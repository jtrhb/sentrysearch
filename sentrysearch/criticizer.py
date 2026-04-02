"""Comprehensive video quality critic for AI-generated content.

Sends video to Gemini multimodal model to identify specific quality
issues across temporal, visual, audio-visual, and compositional dimensions.
Each issue is tagged with category, type, severity, and approximate timestamp.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

CRITIC_MODEL = "gemini-2.5-flash"

# Severity weights for quality grade calculation
SEVERITY_PENALTY = {
    "critical": 25,
    "major": 12,
    "minor": 4,
    "nitpick": 1,
}


class VideoCritic:
    """Identifies specific quality defects in AI-generated videos."""

    def __init__(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    def criticize(self, video_path: str) -> dict:
        """Run comprehensive quality criticism on a video.

        Returns:
            Dict with keys: issues, category_scores, quality_grade, summary.
        """
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

        prompt = _build_critique_prompt()

        t0 = time.monotonic()
        response = self._client.models.generate_content(
            model=CRITIC_MODEL,
            contents=[video_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        elapsed = time.monotonic() - t0
        print(
            f"  [critic] {CRITIC_MODEL} responded in {elapsed:.1f}s",
            file=sys.stderr,
        )

        result = json.loads(response.text)
        return _normalize_result(result)


def _build_critique_prompt() -> str:
    return (
        "You are an expert video quality control critic specializing in "
        "AI-generated content. Your job is to find EVERY flaw, no matter "
        "how small. Be ruthless, thorough, and specific.\n\n"
        "Analyze this video across ALL of the following dimensions and "
        "list every issue you find:\n\n"
        "## 1. Temporal & Motion (temporal)\n"
        "- motion_discontinuity: Sudden jumps, freezes, or unnatural "
        "acceleration/deceleration in movement\n"
        "- physics_violation: Objects defying gravity, impossible "
        "trajectories, unnatural collisions, floating objects\n"
        "- flickering: Temporal instability, objects appearing/disappearing "
        "between frames\n"
        "- motion_blur_artifacts: Unnatural or missing motion blur\n\n"
        "## 2. Visual Artifacts (visual)\n"
        "- hand_deformation: Wrong number of fingers, distorted or fused "
        "fingers, unnatural hand poses\n"
        "- facial_distortion: Feature warping, asymmetry, features shifting "
        "position between frames\n"
        "- texture_swimming: Textures sliding over surfaces, morphing "
        "patterns, unstable surface detail\n"
        "- edge_artifacts: Blurry object boundaries, tearing, aliasing, "
        "halo effects\n"
        "- resolution_inconsistency: Some areas sharp while others are "
        "noticeably blurry or pixelated\n"
        "- text_corruption: Garbled, unreadable, or nonsensical text/signs/"
        "symbols in the scene\n\n"
        "## 3. Character & Object Issues (character)\n"
        "- clipping: Body parts passing through other body parts or objects "
        "(穿模)\n"
        "- proportion_error: Wrong relative sizes of objects, people, or "
        "body parts\n"
        "- appearance_shift: Character face/hair/features changing between "
        "frames\n"
        "- clothing_anomaly: Clothing appearing/disappearing, changing "
        "pattern or color, defying physics\n\n"
        "## 4. Audio-Visual Sync (audio)\n"
        "- lip_sync: Lip/mouth movements don't match speech timing or "
        "phonemes\n"
        "- audio_video_sync: General audio-video timing mismatch\n"
        "- ambient_mismatch: Environmental sounds inconsistent with the "
        "visual scene\n"
        "- audio_artifacts: Noise, distortion, clicks, unnatural cuts, "
        "repeating patterns in audio\n\n"
        "## 5. Composition & Aesthetics (composition)\n"
        "- framing: Awkward composition, subjects cut off, poor visual "
        "balance\n"
        "- color_anomaly: Unnatural colors, color banding, posterization, "
        "oversaturation\n"
        "- lighting_contradiction: Shadows pointing different directions, "
        "inconsistent light sources\n"
        "- depth_of_field: Inconsistent focus planes, unnatural bokeh, "
        "focus shifts\n\n"
        "## 6. Content Coherence (coherence)\n"
        "- spatial_impossibility: Impossible room layouts, contradictory "
        "perspectives, non-Euclidean geometry\n"
        "- style_inconsistency: Mixing different visual styles or art "
        "styles within the same video\n"
        "- logical_error: Events that don't make causal or physical sense\n\n"
        "For EACH issue found, provide:\n"
        "- category: one of temporal, visual, character, audio, composition, "
        "coherence\n"
        "- type: specific issue type from the lists above\n"
        '- severity: "critical" (unwatchable/obvious), "major" (clearly '
        'noticeable), "minor" (visible on close inspection), "nitpick" '
        "(very minor)\n"
        "- description: what specifically is wrong, be precise\n"
        '- timestamp: approximate time range (e.g. "2s-4s") or "throughout" '
        "if persistent\n\n"
        "Also provide:\n"
        "- category_scores: object with 0-100 score for each of the 6 "
        "categories (100 = no issues found in that category)\n"
        "- summary: 2-3 sentence overall quality assessment in Chinese\n\n"
        "If the video has NO audio track, skip the audio category and set "
        "its score to null.\n\n"
        "Respond with JSON only:\n"
        "{\n"
        '  "issues": [{...}, ...],\n'
        '  "category_scores": {"temporal": int, "visual": int, '
        '"character": int, "audio": int|null, "composition": int, '
        '"coherence": int},\n'
        '  "summary": "string"\n'
        "}"
    )


def _normalize_result(raw: dict) -> dict:
    """Normalize and enrich the raw Gemini response."""
    issues = raw.get("issues", [])
    category_scores = raw.get("category_scores", {})
    summary = raw.get("summary", "")

    # Ensure all issues have required fields
    cleaned_issues = []
    for issue in issues:
        cleaned_issues.append(
            {
                "category": issue.get("category", "unknown"),
                "type": issue.get("type", "unknown"),
                "severity": issue.get("severity", "minor"),
                "description": issue.get("description", ""),
                "timestamp": issue.get("timestamp", "unknown"),
            }
        )

    # Count issues by severity
    severity_counts = {"critical": 0, "major": 0, "minor": 0, "nitpick": 0}
    for issue in cleaned_issues:
        sev = issue["severity"]
        if sev in severity_counts:
            severity_counts[sev] += 1

    # Calculate quality grade from severity penalties
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

    # Fill in missing category scores
    all_categories = [
        "temporal",
        "visual",
        "character",
        "audio",
        "composition",
        "coherence",
    ]
    for cat in all_categories:
        if cat not in category_scores:
            category_scores[cat] = 100  # default: no issues

    return {
        "issues": cleaned_issues,
        "category_scores": category_scores,
        "severity_counts": severity_counts,
        "quality_grade": quality_grade,
        "grade_score": grade_score,
        "summary": summary,
    }
