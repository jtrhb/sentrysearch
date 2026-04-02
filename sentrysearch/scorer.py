"""Multi-dimensional video quality scoring using Gemini multimodal models."""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

SCORE_MODEL = "gemini-2.5-flash"


class VideoScorer:
    """Scores videos across multiple quality dimensions using Gemini."""

    def __init__(self):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)

    def _call_gemini(self, video_path: str, prompt: str) -> dict:
        """Send video + prompt to Gemini and parse JSON response."""
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
            f"  [scorer] {SCORE_MODEL} responded in {elapsed:.1f}s",
            file=sys.stderr,
        )

        return json.loads(response.text)

    def score_consistency(self, video_path: str) -> dict:
        """Score character and scene consistency (0-100 each)."""
        prompt = (
            "You are a professional video quality assessor specializing in "
            "AI-generated content evaluation. Analyze this video and rate:\n\n"
            "1. character_consistency (integer 0-100): How consistent are "
            "human/character appearances throughout the video?\n"
            "   - Stable facial features and proportions\n"
            "   - Consistent clothing and accessories\n"
            "   - Smooth, natural body movements\n"
            "   - No sudden appearance changes or morphing\n"
            "   - If no characters are present, score based on main subject consistency\n\n"
            "2. scene_consistency (integer 0-100): How consistent is the "
            "environment throughout?\n"
            "   - Stable background elements\n"
            "   - Consistent lighting direction and color temperature\n"
            "   - No sudden spatial inconsistencies\n"
            "   - Consistent perspective and camera movement\n\n"
            "3. consistency_notes: Brief explanation of any issues found.\n\n"
            "Respond with JSON only:\n"
            '{"character_consistency": <int>, "scene_consistency": <int>, '
            '"consistency_notes": "<string>"}'
        )
        result = self._call_gemini(video_path, prompt)
        return {
            "character_consistency": int(result.get("character_consistency", 50)),
            "scene_consistency": int(result.get("scene_consistency", 50)),
            "consistency_notes": result.get("consistency_notes", ""),
        }

    def score_ai_detection(self, video_path: str) -> dict:
        """Score how AI-generated the video appears (0-100)."""
        prompt = (
            "You are an expert at detecting AI-generated video content. "
            "Analyze this video and rate how AI-generated it appears.\n\n"
            "ai_score (integer 0-100):\n"
            "  0 = Clearly real/natural footage, no AI artifacts\n"
            "  30 = Minor suspicious elements but mostly natural\n"
            "  60 = Noticeable AI artifacts, likely AI-generated\n"
            "  80 = Obvious AI generation with clear artifacts\n"
            "  100 = Unmistakably AI-generated\n\n"
            "Look for these common AI video artifacts:\n"
            "- Unnatural hand/finger deformation\n"
            "- Facial feature warping or flickering\n"
            "- Inconsistent lighting or shadows between frames\n"
            "- Texture swimming or morphing\n"
            "- Unnatural motion (too smooth or jittery)\n"
            "- Background warping or instability\n"
            "- Object deformation during movement\n"
            "- Temporal inconsistencies between frames\n"
            "- Unnatural hair or fabric physics\n\n"
            "Respond with JSON only:\n"
            '{"ai_score": <int>, "artifacts": ["<artifact1>", ...], '
            '"ai_notes": "<string>"}'
        )
        result = self._call_gemini(video_path, prompt)
        return {
            "ai_score": int(result.get("ai_score", 50)),
            "artifacts": result.get("artifacts", []),
            "ai_notes": result.get("ai_notes", ""),
        }

    def score_similarity(self, video_path: str, store) -> dict:
        """Check similarity against existing indexed assets.

        Embeds the video and queries pgvector for nearest neighbors.
        """
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

    def score_video(
        self,
        video_path: str,
        store=None,
        weights: dict | None = None,
    ) -> dict:
        """Run full multi-dimensional scoring on a video.

        Args:
            video_path: Path to the video file.
            store: SentryStore instance for similarity check. If None or
                   the index is empty, similarity scoring is skipped.
            weights: Custom weights for overall score calculation.
                     Keys: consistency, ai, similarity. Must sum to 1.0.

        Returns:
            Dict with per-dimension scores, details, and overall score.
        """
        if weights is None:
            weights = {
                "consistency": 0.35,
                "ai": 0.40,
                "similarity": 0.25,
            }

        consistency = self.score_consistency(video_path)
        ai_detection = self.score_ai_detection(video_path)

        # Similarity check: skip if no store or index is empty
        similarity = {"max_similarity": 0.0, "similar_to": None, "top_matches": []}
        has_index = False
        if store is not None:
            stats = store.get_stats()
            has_index = stats["total_chunks"] > 0
            if has_index:
                similarity = self.score_similarity(video_path, store)

        # Compute per-dimension normalized scores (0-100)
        consistency_avg = (
            consistency["character_consistency"]
            + consistency["scene_consistency"]
        ) / 2

        ai_score_raw = ai_detection["ai_score"]
        ai_quality = 100 - ai_score_raw  # lower AI = better

        originality = (1 - similarity["max_similarity"]) * 100

        # If no indexed assets exist, redistribute similarity weight
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

        overall = (
            consistency_avg * effective_weights["consistency"]
            + ai_quality * effective_weights["ai"]
            + originality * effective_weights["similarity"]
        )

        return {
            "consistency": {
                "character": consistency["character_consistency"],
                "scene": consistency["scene_consistency"],
                "average": round(consistency_avg, 1),
                "notes": consistency["consistency_notes"],
            },
            "ai_detection": {
                "score": ai_score_raw,
                "quality": round(ai_quality, 1),
                "artifacts": ai_detection["artifacts"],
                "notes": ai_detection["ai_notes"],
            },
            "similarity": similarity,
            "overall_score": round(overall, 1),
        }
