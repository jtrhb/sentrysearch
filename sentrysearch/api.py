"""FastAPI web API for SentrySearch."""

import os
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

# Maximum video duration (seconds) Gemini Embedding 2 accepts in a single request.
MAX_SINGLE_EMBED_DURATION = 120

app = FastAPI(
    title="SentrySearch",
    description="Semantic search over video footage via natural language queries.",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    r2_keys: list[str]
    preprocess: bool = True
    target_resolution: int = 480
    target_fps: int = 5
    # Chunking params — only used when a video exceeds 120s
    chunk_duration: int = 30
    overlap: int = 5


class IndexResponse(BaseModel):
    new_videos: int
    new_chunks: int
    total_chunks: int
    total_files: int


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    threshold: float = 0.41


class SearchResult(BaseModel):
    source_file: str
    start_time: float
    end_time: float
    similarity_score: float
    video_url: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class StatsResponse(BaseModel):
    total_chunks: int
    unique_source_files: int
    source_files: list[str]


class ScoreRequest(BaseModel):
    r2_keys: list[str]
    check_similarity: bool = True
    weights: dict | None = None


class VideoScoreResult(BaseModel):
    source_file: str
    consistency: dict
    ai_detection: dict
    similarity: dict
    overall_score: float


class ScoreResponse(BaseModel):
    scores: list[VideoScoreResult]


class CritiqueRequest(BaseModel):
    r2_keys: list[str]


class CritiqueIssue(BaseModel):
    category: str
    type: str
    severity: str
    description: str
    timestamp: str


class CritiqueResult(BaseModel):
    source_file: str
    issues: list[CritiqueIssue]
    category_scores: dict
    severity_counts: dict
    quality_grade: str
    grade_score: float
    summary: str


class CritiqueResponse(BaseModel):
    critiques: list[CritiqueResult]


class RemoveRequest(BaseModel):
    source_files: list[str]


class RemoveResponse(BaseModel):
    removed_chunks: int


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def _get_store():
    from .store import SentryStore

    return SentryStore()


def _get_r2():
    from .r2 import R2Client

    return R2Client()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    store = _get_store()
    try:
        return store.get_stats()
    finally:
        store.close()


@app.post("/index", response_model=IndexResponse)
def index_videos(request: IndexRequest):
    """Index video files from R2.

    Videos ≤120s are embedded as a whole.  Longer videos are automatically
    split into overlapping chunks before embedding.
    """
    from .chunker import (
        _get_video_duration,
        chunk_video,
        is_still_frame_chunk,
        preprocess_chunk,
    )
    from .embedder import get_embedder, reset_embedder

    store = _get_store()
    r2 = _get_r2()

    try:
        embedder = get_embedder("gemini")
        new_videos = 0
        new_chunks = 0

        for r2_key in request.r2_keys:
            if store.is_indexed(r2_key):
                continue

            local_path = r2.download_temp(r2_key)
            tmp_dir = None

            try:
                duration = _get_video_duration(local_path)

                if duration <= MAX_SINGLE_EMBED_DURATION:
                    # --- Whole-video embedding ---
                    embed_path = local_path
                    if request.preprocess:
                        embed_path = preprocess_chunk(
                            local_path,
                            target_resolution=request.target_resolution,
                            target_fps=request.target_fps,
                        )

                    embedding = embedder.embed_video_chunk(embed_path)

                    store.add_chunks(
                        [
                            {
                                "source_file": r2_key,
                                "start_time": 0.0,
                                "end_time": duration,
                                "embedding": embedding,
                            }
                        ]
                    )
                    new_videos += 1
                    new_chunks += 1

                    if embed_path != local_path:
                        try:
                            os.unlink(embed_path)
                        except OSError:
                            pass

                else:
                    # --- Chunked embedding for long videos ---
                    chunks = chunk_video(
                        local_path,
                        chunk_duration=request.chunk_duration,
                        overlap=request.overlap,
                    )
                    if chunks:
                        tmp_dir = os.path.dirname(chunks[0]["chunk_path"])

                    embedded = []
                    files_to_cleanup = []

                    for chunk in chunks:
                        if is_still_frame_chunk(chunk["chunk_path"]):
                            files_to_cleanup.append(chunk["chunk_path"])
                            continue

                        embed_path = chunk["chunk_path"]
                        if request.preprocess:
                            embed_path = preprocess_chunk(
                                embed_path,
                                target_resolution=request.target_resolution,
                                target_fps=request.target_fps,
                            )
                            if embed_path != chunk["chunk_path"]:
                                files_to_cleanup.append(embed_path)

                        embedding = embedder.embed_video_chunk(embed_path)
                        embedded.append(
                            {
                                "source_file": r2_key,
                                "start_time": chunk["start_time"],
                                "end_time": chunk["end_time"],
                                "embedding": embedding,
                            }
                        )
                        files_to_cleanup.append(chunk["chunk_path"])

                    for f in files_to_cleanup:
                        try:
                            os.unlink(f)
                        except OSError:
                            pass

                    if embedded:
                        store.add_chunks(embedded)
                        new_videos += 1
                        new_chunks += len(embedded)

            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

        s = store.get_stats()
        return IndexResponse(
            new_videos=new_videos,
            new_chunks=new_chunks,
            total_chunks=s["total_chunks"],
            total_files=s["unique_source_files"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_embedder()
        store.close()


@app.post("/search", response_model=SearchResponse)
def search_footage(request: SearchRequest):
    """Search indexed footage with a natural language query.

    Returns matching videos with presigned R2 URLs for direct playback.
    """
    from .embedder import get_embedder, reset_embedder
    from .search import search_footage as _search

    store = _get_store()

    try:
        if store.get_stats()["total_chunks"] == 0:
            raise HTTPException(
                status_code=404,
                detail="No indexed footage. POST /index first.",
            )

        get_embedder("gemini")

        results = _search(request.query, store, n_results=request.n_results)

        if not results:
            return SearchResponse(results=[])

        r2 = _get_r2()
        output = []

        for r in results:
            video_url = None
            if r["similarity_score"] >= request.threshold:
                video_url = r2.presigned_url(r["source_file"])

            output.append(
                SearchResult(
                    source_file=r["source_file"],
                    start_time=r["start_time"],
                    end_time=r["end_time"],
                    similarity_score=r["similarity_score"],
                    video_url=video_url,
                )
            )

        return SearchResponse(results=output)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_embedder()
        store.close()


@app.post("/score", response_model=ScoreResponse)
def score_videos(request: ScoreRequest):
    """Score video files from R2 across multiple quality dimensions.

    Evaluates character/scene consistency, AI-generated artifacts, and
    similarity to existing indexed assets.
    """
    from .embedder import reset_embedder
    from .scorer import VideoScorer

    store = _get_store() if request.check_similarity else None
    r2 = _get_r2()
    scorer = VideoScorer()

    try:
        scores = []
        for r2_key in request.r2_keys:
            local_path = r2.download_temp(r2_key)
            try:
                result = scorer.score_video(
                    local_path,
                    store=store,
                    weights=request.weights,
                )
                result["source_file"] = r2_key

                if store is not None:
                    store.save_score(r2_key, result)

                scores.append(result)
            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

        return ScoreResponse(scores=scores)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_embedder()
        if store is not None:
            store.close()


@app.get("/scores")
def get_scores(
    min_overall: float | None = None,
    max_ai_score: float | None = None,
    limit: int = 50,
):
    """Query stored video scores with optional filters."""
    store = _get_store()
    try:
        return store.get_scores(
            min_overall=min_overall,
            max_ai_score=max_ai_score,
            limit=limit,
        )
    finally:
        store.close()


@app.get("/scores/{source_file:path}")
def get_score(source_file: str):
    """Get scores for a specific video."""
    store = _get_store()
    try:
        result = store.get_score(source_file)
        if result is None:
            raise HTTPException(status_code=404, detail="No scores found")
        return result
    finally:
        store.close()


@app.post("/critique", response_model=CritiqueResponse)
def critique_videos(request: CritiqueRequest):
    """Run comprehensive quality criticism on videos from R2.

    Identifies specific defects across temporal, visual, audio-visual,
    compositional, and coherence dimensions with severity ratings.
    """
    from .criticizer import VideoCritic

    store = _get_store()
    r2 = _get_r2()
    critic = VideoCritic()

    try:
        critiques = []
        for r2_key in request.r2_keys:
            local_path = r2.download_temp(r2_key)
            try:
                result = critic.criticize(local_path)
                result["source_file"] = r2_key
                store.save_critique(r2_key, result)
                critiques.append(result)
            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

        return CritiqueResponse(critiques=critiques)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        store.close()


@app.get("/critiques")
def get_critiques(
    max_grade: str | None = None,
    limit: int = 50,
):
    """Query stored video critiques. Sorted by worst quality first."""
    store = _get_store()
    try:
        return store.get_critiques(max_grade=max_grade, limit=limit)
    finally:
        store.close()


@app.get("/critiques/{source_file:path}")
def get_critique(source_file: str):
    """Get critique for a specific video."""
    store = _get_store()
    try:
        result = store.get_critique(source_file)
        if result is None:
            raise HTTPException(status_code=404, detail="No critique found")
        return result
    finally:
        store.close()


@app.post("/remove", response_model=RemoveResponse)
def remove_files(request: RemoveRequest):
    """Remove specific files from the index."""
    store = _get_store()
    try:
        total = 0
        s = store.get_stats()
        for pattern in request.source_files:
            matches = [f for f in s["source_files"] if pattern in f]
            for source_file in matches:
                total += store.remove_file(source_file)
        return RemoveResponse(removed_chunks=total)
    finally:
        store.close()


@app.post("/reset")
def reset_index():
    """Delete all indexed data."""
    store = _get_store()
    try:
        s = store.get_stats()
        for f in s["source_files"]:
            store.remove_file(f)
        return {
            "removed_chunks": s["total_chunks"],
            "removed_files": s["unique_source_files"],
        }
    finally:
        store.close()
