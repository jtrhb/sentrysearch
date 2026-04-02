"""Unified video evaluation: multi-dimensional scoring + comprehensive defect detection.

Merges the functionality of scorer.py (quality scoring) and criticizer.py (defect
detection) into a single Gemini API call, then enriches the result with pgvector
similarity and severity-based grading.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

SCORE_MODEL = "gemini-2.5-flash"

# Severity weights for quality grade calculation
SEVERITY_PENALTY = {
    "critical": 25,
    "major": 12,
    "minor": 4,
    "nitpick": 1,
}

# ── Defect taxonomy ──────────────────────────────────────────────────────────
# 6 categories, 25 types
DEFECT_TAXONOMY = {
    "temporal": [
        "motion_discontinuity",
        "physics_violation",
        "flickering",
        "motion_blur_artifacts",
    ],
    "visual": [
        "hand_deformation",
        "facial_distortion",
        "texture_swimming",
        "edge_artifacts",
        "resolution_inconsistency",
        "text_corruption",
    ],
    "character": [
        "clipping",
        "proportion_error",
        "appearance_shift",
        "clothing_anomaly",
    ],
    "audio": [
        "lip_sync",
        "audio_video_sync",
        "ambient_mismatch",
        "audio_artifacts",
    ],
    "composition": [
        "framing",
        "color_anomaly",
        "lighting_contradiction",
        "depth_of_field",
    ],
    "coherence": [
        "spatial_impossibility",
        "style_inconsistency",
        "logical_error",
    ],
}


class VideoEvaluator:
    """Scores and critiques AI-generated videos in a single Gemini call."""

    def __init__(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    # ── public entry point ───────────────────────────────────────────────

    def evaluate(
        self,
        video_path: str,
        store=None,
        weights: dict | None = None,
    ) -> dict:
        """Run full evaluation (scoring + criticism) on a video.

        Args:
            video_path: Path to the video file.
            store: SentryStore instance for similarity check.  If *None* or
                   the index is empty, similarity scoring is skipped.
            weights: Custom weights for the overall score.
                     Keys: consistency, ai, similarity.  Must sum to 1.0.

        Returns:
            Dict with scores, category_scores, issues, quality_grade,
            similarity info, and a Chinese-language summary.
        """
        if weights is None:
            weights = {
                "consistency": 0.35,
                "ai": 0.40,
                "similarity": 0.25,
            }

        # 1. Single Gemini call: scoring + criticism
        gemini_result = self._call_gemini(video_path)

        # 2. Similarity via pgvector (separate path)
        similarity = {"max_similarity": 0.0, "similar_to": None, "top_matches": []}
        has_index = False
        if store is not None:
            stats = store.get_stats()
            has_index = stats["total_chunks"] > 0
            if has_index:
                similarity = self._score_similarity(video_path, store)

        # 3. Derive composite scores
        return self._build_result(gemini_result, similarity, has_index, weights)

    # ── Gemini call ──────────────────────────────────────────────────────

    def _call_gemini(self, video_path: str) -> dict:
        """Send video + combined prompt to Gemini; parse JSON response."""
        from google.genai import types

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        if hasattr(types.Part, "from_bytes"):
            video_part = types.Part.from_bytes(
                data=video_bytes, mime_type="video/mp4"
            )
        else:
            video_part = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
            )

        prompt = _build_evaluation_prompt()

        t0 = time.monotonic()
        response = self._client.models.generate_content(
            model=SCORE_MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        elapsed = time.monotonic() - t0
        print(
            f"  [evaluator] {SCORE_MODEL} responded in {elapsed:.1f}s",
            file=sys.stderr,
        )

        return json.loads(response.text)

    # ── Similarity via pgvector ──────────────────────────────────────────

    @staticmethod
    def _score_similarity(video_path: str, store) -> dict:
        """Embed video and query pgvector for nearest neighbours."""
        from .embedder import get_embedder

        embedder = get_embedder("gemini")
        embedding = embedder.embed_video_chunk(video_path)

        hits = store.search(embedding, n_results=3)
        if hits and hits[0]["score"] > 0:
            return {
                "max_similarity": round(hits[0]["score"], 4),
                "similar_to": hits[0]["source_file"],
                "top_matches": [
                    {
                        "source_file": h["source_file"],
                        "similarity": round(h["score"], 4),
                    }
                    for h in hits
                ],
            }
        return {
            "max_similarity": 0.0,
            "similar_to": None,
            "top_matches": [],
        }

    # ── Result assembly ──────────────────────────────────────────────────

    @staticmethod
    def _build_result(
        gemini: dict,
        similarity: dict,
        has_index: bool,
        weights: dict,
    ) -> dict:
        """Merge Gemini output, similarity data, and severity grading."""

        # ── Extract Gemini fields ────────────────────────────────────────
        character_consistency = int(gemini.get("character_consistency", 50))
        scene_consistency = int(gemini.get("scene_consistency", 50))
        ai_score = int(gemini.get("ai_score", 50))
        raw_category_scores = gemini.get("category_scores", {})
        raw_issues = gemini.get("issues", [])
        summary = gemini.get("summary", "")

        # ── Normalise category scores ────────────────────────────────────
        all_categories = [
            "temporal",
            "visual",
            "character",
            "audio",
            "composition",
            "coherence",
        ]
        category_scores = {}
        for cat in all_categories:
            if cat not in raw_category_scores:
                category_scores[cat] = 100
            else:
                category_scores[cat] = raw_category_scores[cat]

        # ── Clean issues ─────────────────────────────────────────────────
        cleaned_issues = []
        for issue in raw_issues:
            cleaned_issues.append(
                {
                    "category": issue.get("category", "unknown"),
                    "type": issue.get("type", "unknown"),
                    "severity": issue.get("severity", "minor"),
                    "description": issue.get("description", ""),
                    "timestamp": issue.get("timestamp", "unknown"),
                }
            )

        # ── Severity counts & quality grade ──────────────────────────────
        severity_counts = {"critical": 0, "major": 0, "minor": 0, "nitpick": 0}
        for issue in cleaned_issues:
            sev = issue["severity"]
            if sev in severity_counts:
                severity_counts[sev] += 1

        total_penalty = sum(
            severity_counts[sev] * SEVERITY_PENALTY[sev] for sev in severity_counts
        )
        grade_score = max(0, 100 - total_penalty)

        if grade_score >= 90:
            quality_grade = "A"
        elif grade_score >= 75:
            quality_grade = "B"
        elif grade_score >= 60:
            quality_grade = "C"
        elif grade_score >= 40:
            quality_grade = "D"
        else:
            quality_grade = "F"

        # ── Overall score (weighted) ─────────────────────────────────────
        consistency_avg = (character_consistency + scene_consistency) / 2
        ai_quality = 100 - ai_score  # lower AI-ness = better quality
        originality = (1 - similarity["max_similarity"]) * 100

        if not has_index:
            effective_weights = {
                "consistency": weights["consistency"]
                / (weights["consistency"] + weights["ai"]),
                "ai": weights["ai"]
                / (weights["consistency"] + weights["ai"]),
                "similarity": 0.0,
            }
        else:
            effective_weights = weights

        overall_score = (
            consistency_avg * effective_weights["consistency"]
            + ai_quality * effective_weights["ai"]
            + originality * effective_weights["similarity"]
        )

        # ── Assemble final dict ──────────────────────────────────────────
        return {
            "consistency": {
                "character": character_consistency,
                "scene": scene_consistency,
                "average": round(consistency_avg, 1),
            },
            "ai_detection": {
                "score": ai_score,
                "quality": round(ai_quality, 1),
            },
            "similarity": similarity,
            "category_scores": category_scores,
            "issues": cleaned_issues,
            "severity_counts": severity_counts,
            "quality_grade": quality_grade,
            "grade_score": grade_score,
            "overall_score": round(overall_score, 1),
            "summary": summary,
        }


# ── Prompt construction ──────────────────────────────────────────────────────


def _build_evaluation_prompt() -> str:
    """Build the combined scoring + criticism prompt for Gemini."""
    return (
        "You are an expert video quality assessor AND a ruthless quality "
        "control critic specializing in AI-generated content. Perform a "
        "SINGLE comprehensive evaluation that covers both scoring and defect "
        "detection.\n\n"
        # ── Part A: Scoring ──────────────────────────────────────────────
        "## PART A — Quality Scores\n\n"
        "Rate each of the following on an integer scale 0-100:\n\n"
        "1. **character_consistency** — How consistent are human/character "
        "appearances throughout the video?\n"
        "   - Stable facial features and proportions\n"
        "   - Consistent clothing and accessories\n"
        "   - Smooth, natural body movements\n"
        "   - No sudden appearance changes or morphing\n"
        "   - If no characters are present, score based on main subject "
        "consistency\n\n"
        "2. **scene_consistency** — How consistent is the environment?\n"
        "   - Stable background elements\n"
        "   - Consistent lighting direction and color temperature\n"
        "   - No sudden spatial inconsistencies\n"
        "   - Consistent perspective and camera movement\n\n"
        "3. **ai_score** — How AI-generated does the video appear?\n"
        "   0 = Clearly real/natural footage, no AI artifacts\n"
        "   30 = Minor suspicious elements but mostly natural\n"
        "   60 = Noticeable AI artifacts, likely AI-generated\n"
        "   80 = Obvious AI generation with clear artifacts\n"
        "   100 = Unmistakably AI-generated\n\n"
        "   Look for: unnatural hand/finger deformation, facial warping, "
        "inconsistent lighting/shadows, texture swimming, unnatural motion, "
        "background warping, object deformation, temporal inconsistencies, "
        "unnatural hair/fabric physics.\n\n"
        # ── Part B: Defect Detection ─────────────────────────────────────
        "## PART B — Defect Detection\n\n"
        "Find EVERY flaw, no matter how small. Be ruthless, thorough, and "
        "specific. Analyze across ALL of the following dimensions:\n\n"
        "### 1. Temporal & Motion (temporal)\n"
        "- motion_discontinuity: Sudden jumps, freezes, or unnatural "
        "acceleration/deceleration\n"
        "- physics_violation: Objects defying gravity, impossible "
        "trajectories, floating objects\n"
        "- flickering: Temporal instability, objects appearing/disappearing "
        "between frames\n"
        "- motion_blur_artifacts: Unnatural or missing motion blur\n\n"
        "### 2. Visual Artifacts (visual)\n"
        "- hand_deformation: Wrong number of fingers, distorted hands\n"
        "- facial_distortion: Feature warping, asymmetry, shifting features\n"
        "- texture_swimming: Textures sliding over surfaces, morphing "
        "patterns\n"
        "- edge_artifacts: Blurry boundaries, tearing, aliasing, halos\n"
        "- resolution_inconsistency: Sharp areas mixed with blurry/pixelated "
        "areas\n"
        "- text_corruption: Garbled, unreadable text/signs/symbols\n\n"
        "### 3. Character & Object Issues (character)\n"
        "- clipping: Body parts passing through other objects\n"
        "- proportion_error: Wrong relative sizes\n"
        "- appearance_shift: Face/hair/features changing between frames\n"
        "- clothing_anomaly: Clothing appearing/disappearing or changing\n\n"
        "### 4. Audio-Visual Sync (audio)\n"
        "- lip_sync: Mouth movements don't match speech\n"
        "- audio_video_sync: General audio-video timing mismatch\n"
        "- ambient_mismatch: Environmental sounds inconsistent with scene\n"
        "- audio_artifacts: Noise, distortion, clicks, repeating patterns\n\n"
        "### 5. Composition & Aesthetics (composition)\n"
        "- framing: Awkward composition, subjects cut off\n"
        "- color_anomaly: Unnatural colors, banding, posterization\n"
        "- lighting_contradiction: Shadows in different directions\n"
        "- depth_of_field: Inconsistent focus, unnatural bokeh\n\n"
        "### 6. Content Coherence (coherence)\n"
        "- spatial_impossibility: Impossible layouts, non-Euclidean geometry\n"
        "- style_inconsistency: Mixed visual/art styles\n"
        "- logical_error: Events that don't make causal sense\n\n"
        "For EACH issue provide:\n"
        '- category: one of temporal, visual, character, audio, composition, '
        "coherence\n"
        "- type: specific issue type from the lists above\n"
        '- severity: "critical" (unwatchable), "major" (clearly noticeable), '
        '"minor" (visible on close inspection), "nitpick" (very minor)\n'
        "- description: what specifically is wrong — be precise\n"
        '- timestamp: approximate time range (e.g. "2s-4s") or "throughout"\n\n'
        "Also provide:\n"
        "- **category_scores**: object with a 0-100 quality score for each of "
        "the 6 categories (100 = no issues). If no audio track, set audio to "
        "null.\n"
        "- **summary**: 2-3 sentence overall quality assessment **in Chinese**.\n\n"
        # ── Output schema ────────────────────────────────────────────────
        "## Output\n\n"
        "Respond with JSON only — no markdown fences:\n"
        "{\n"
        '  "character_consistency": <int 0-100>,\n'
        '  "scene_consistency": <int 0-100>,\n'
        '  "ai_score": <int 0-100>,\n'
        '  "category_scores": {\n'
        '    "temporal": <int>, "visual": <int>, "character": <int>,\n'
        '    "audio": <int|null>, "composition": <int>, "coherence": <int>\n'
        "  },\n"
        '  "issues": [\n'
        "    {\n"
        '      "category": "<string>",\n'
        '      "type": "<string>",\n'
        '      "severity": "<critical|major|minor|nitpick>",\n'
        '      "description": "<string>",\n'
        '      "timestamp": "<string>"\n'
        "    }\n"
        "  ],\n"
        '  "summary": "<string in Chinese>"\n'
        "}"
    )
