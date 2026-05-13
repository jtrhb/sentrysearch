"""FastAPI web API for SentrySearch."""

import base64
import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

MAX_SINGLE_EMBED_DURATION = 120
API_KEY = os.environ.get("API_KEY")

app = FastAPI(
    title="SentrySearch",
    description="Semantic search, evaluation, and asset management for AI-generated video.",
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if API_KEY and request.url.path != "/health":
        # Accept either header style: X-API-Key (legacy) or
        # Authorization: Bearer <key> (OpenAI / ChatCut convention).
        key = request.headers.get("X-API-Key")
        if not key:
            auth = request.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                key = auth[7:].strip()
        if key != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    r2_keys: list[str]
    preprocess: bool = True
    target_resolution: int = 480
    target_fps: int = 5
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
    asset_id: str | None = None
    source_file: str
    start_time: float
    end_time: float
    similarity_score: float
    current_path: str | None = None
    video_url: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class EvaluateRequest(BaseModel):
    r2_keys: list[str]
    check_similarity: bool = True
    weights: dict | None = None


class EvaluateResponse(BaseModel):
    evaluations: list[dict]


class ReconcileRequest(BaseModel):
    asset_id: str
    new_path: str


class ReconcileResponse(BaseModel):
    found: bool
    asset_id: str
    new_path: str


class StaleRequest(BaseModel):
    path: str


class StatsResponse(BaseModel):
    total_chunks: int
    unique_source_files: int
    source_files: list[str]


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
# Health / Stats
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


# ---------------------------------------------------------------------------
# /v1/embeddings — OpenAI-shape endpoint over gemini-embedding-2-preview.
#
# Used by external services (e.g. ChatCut agent) that want a unified
# multimodal embedding service without re-implementing Gemini auth and
# rate-limiting. Supports text inputs today; image/video data URIs are
# accepted via the OpenAI content-block shape and routed through
# Part.from_bytes for the same vector space.
#
# Request:
#   { "model": "gemini-embedding-2", "input": str | [str] | [content_block],
#     "dimensions": 768 }
# Content block (multimodal):
#   { "type": "image_url", "image_url": "data:image/png;base64,..." }
#   { "type": "video_url", "video_url": "data:video/mp4;base64,..." }
#   { "type": "text",      "text": "..." }
# Response (OpenAI shape):
#   { "object": "list",
#     "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
#     "model": "gemini-embedding-2-preview" }
# ---------------------------------------------------------------------------


class EmbeddingsRequest(BaseModel):
    input: str | list  # str | list[str] | list[dict]
    model: str | None = None  # accepted but ignored — always gemini-embedding-2-preview
    dimensions: int | None = None
    task_type: str | None = None  # "RETRIEVAL_DOCUMENT" (default) | "RETRIEVAL_QUERY"


def _decode_data_uri(uri: str) -> tuple[bytes, str]:
    """Decode a `data:<mime>;base64,<payload>` URI to (bytes, mime_type)."""
    if not uri.startswith("data:"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected data: URI, got {uri[:32]}...",
        )
    head, _, payload = uri[5:].partition(",")
    if ";base64" not in head:
        raise HTTPException(
            status_code=400,
            detail="Only base64-encoded data: URIs are supported",
        )
    mime = head.split(";")[0] or "application/octet-stream"
    try:
        data = base64.b64decode(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}")
    return data, mime


def _embed_one(item, dims: int, task_type: str) -> list[float]:
    """Embed a single input item (str | content block dict) and return a vector
    of length `dims` (MRL truncation handled by Gemini's output_dimensionality).
    """
    from google.genai import types

    from .embedder import get_embedder

    embedder = get_embedder("gemini")
    client = embedder._client
    limiter = embedder._limiter

    # ---- Build the Gemini content parts ----
    if isinstance(item, str):
        contents = item
    elif isinstance(item, dict):
        item_type = item.get("type")
        if item_type == "text":
            contents = item.get("text") or ""
        elif item_type == "image_url":
            url = item.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="image_url missing url")
            data, mime = _decode_data_uri(url)
            part = (
                types.Part.from_bytes(data=data, mime_type=mime)
                if hasattr(types.Part, "from_bytes")
                else types.Part(inline_data=types.Blob(data=data, mime_type=mime))
            )
            contents = types.Content(parts=[part])
        elif item_type == "video_url":
            url = item.get("video_url")
            if isinstance(url, dict):
                url = url.get("url")
            if not url:
                raise HTTPException(status_code=400, detail="video_url missing url")
            data, mime = _decode_data_uri(url)
            part = (
                types.Part.from_bytes(data=data, mime_type=mime)
                if hasattr(types.Part, "from_bytes")
                else types.Part(inline_data=types.Blob(data=data, mime_type=mime))
            )
            contents = types.Content(parts=[part])
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content block type: {item_type!r}",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Each input must be a string or content block dict, got {type(item).__name__}",
        )

    # ---- Call Gemini ----
    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=dims,
    )

    from .gemini_embedder import _retry

    limiter.wait()
    response = _retry(
        lambda: client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=contents,
            config=config,
        )
    )
    return response.embeddings[0].values


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest):
    """OpenAI-shape multimodal embedding endpoint backed by Gemini.

    Designed for clients that already speak the OpenAI /v1/embeddings
    contract (e.g. ChatCut agent's EmbeddingClient).
    """
    inputs: list = req.input if isinstance(req.input, list) else [req.input]
    if not inputs:
        raise HTTPException(status_code=400, detail="`input` is required")
    dims = req.dimensions or 3072
    if dims < 1 or dims > 3072:
        raise HTTPException(
            status_code=400,
            detail="dimensions must be in [1, 3072]",
        )
    task_type = req.task_type or "RETRIEVAL_DOCUMENT"

    try:
        vectors = [_embed_one(item, dims, task_type) for item in inputs]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": v, "index": i}
            for i, v in enumerate(vectors)
        ],
        "model": "gemini-embedding-2-preview",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


# ---------------------------------------------------------------------------
# Index — with asset registration (ETag as asset_id)
# ---------------------------------------------------------------------------


@app.post("/index", response_model=IndexResponse)
def index_videos(request: IndexRequest):
    """Index video files from R2.

    Creates an asset record (keyed by R2 ETag) and embeds the video.
    Videos ≤120s are embedded whole; longer ones are auto-chunked.
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
            # Get ETag as asset_id
            etag = r2.get_etag(r2_key)

            # Skip if already indexed
            if store.is_indexed(etag):
                continue

            # Register asset
            filename = os.path.basename(r2_key)
            store.register_asset(
                asset_id=etag,
                current_path=r2_key,
                original_path=r2_key,
                filename=filename,
            )

            local_path = r2.download_temp(r2_key)
            tmp_dir = None

            try:
                duration = _get_video_duration(local_path)

                if duration <= MAX_SINGLE_EMBED_DURATION:
                    embed_path = local_path
                    if request.preprocess:
                        embed_path = preprocess_chunk(
                            local_path,
                            target_resolution=request.target_resolution,
                            target_fps=request.target_fps,
                        )

                    embedding = embedder.embed_video_chunk(embed_path)
                    store.add_chunks([{
                        "source_file": etag,
                        "start_time": 0.0,
                        "end_time": duration,
                        "embedding": embedding,
                    }])
                    new_videos += 1
                    new_chunks += 1

                    if embed_path != local_path:
                        try:
                            os.unlink(embed_path)
                        except OSError:
                            pass
                else:
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
                        embedded.append({
                            "source_file": etag,
                            "start_time": chunk["start_time"],
                            "end_time": chunk["end_time"],
                            "embedding": embedding,
                        })
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


# ---------------------------------------------------------------------------
# Search — resolves asset_id → current_path → presigned URL
# ---------------------------------------------------------------------------


@app.post("/search", response_model=SearchResponse)
def search_footage(request: SearchRequest):
    """Search indexed footage with a natural language query.

    Returns matching videos with current R2 paths and presigned URLs.
    The source_file in chunks is an asset_id (ETag); this endpoint
    resolves it to the current path via the assets table.
    """
    from .embedder import get_embedder, reset_embedder
    from .search import search_footage as _search

    store = _get_store()

    try:
        if store.get_stats()["total_chunks"] == 0:
            raise HTTPException(status_code=404, detail="No indexed footage. POST /index first.")

        get_embedder("gemini")
        results = _search(request.query, store, n_results=request.n_results)

        if not results:
            return SearchResponse(results=[])

        r2 = _get_r2()
        output = []

        for r in results:
            asset_id = r["source_file"]  # This is now an ETag
            asset = store.get_asset(asset_id)

            current_path = asset["current_path"] if asset else None
            video_url = None
            if current_path and r["similarity_score"] >= request.threshold:
                video_url = r2.presigned_url(current_path)

            output.append(SearchResult(
                asset_id=asset_id,
                source_file=current_path or asset_id,
                start_time=r["start_time"],
                end_time=r["end_time"],
                similarity_score=r["similarity_score"],
                current_path=current_path,
                video_url=video_url,
            ))

        return SearchResponse(results=output)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_embedder()
        store.close()


# ---------------------------------------------------------------------------
# Evaluate — unified scoring + criticism
# ---------------------------------------------------------------------------


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_videos(request: EvaluateRequest):
    """Run comprehensive quality evaluation on videos from R2.

    Single Gemini call per video covering consistency scoring, AI detection,
    and defect detection across 6 categories (25 defect types).
    """
    from .embedder import reset_embedder
    from .evaluator import VideoEvaluator

    store = _get_store() if request.check_similarity else None
    r2 = _get_r2()
    evaluator = VideoEvaluator()

    try:
        evaluations = []
        for r2_key in request.r2_keys:
            etag = r2.get_etag(r2_key)
            local_path = r2.download_temp(r2_key)
            try:
                result = evaluator.evaluate(
                    local_path,
                    store=store,
                    weights=request.weights,
                )
                result["asset_id"] = etag
                result["source_file"] = r2_key

                if store is not None:
                    store.save_evaluation(etag, result)

                evaluations.append(result)
            finally:
                try:
                    os.unlink(local_path)
                except OSError:
                    pass

        return EvaluateResponse(evaluations=evaluations)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        reset_embedder()
        if store is not None:
            store.close()


@app.get("/evaluations")
def get_evaluations(
    min_overall: float | None = None,
    max_ai_score: float | None = None,
    max_grade: str | None = None,
    limit: int = 50,
):
    """Query stored evaluations with optional filters."""
    store = _get_store()
    try:
        return store.get_evaluations(
            min_overall=min_overall,
            max_ai_score=max_ai_score,
            max_grade=max_grade,
            limit=limit,
        )
    finally:
        store.close()


@app.get("/evaluations/{asset_id}")
def get_evaluation(asset_id: str):
    """Get evaluation for a specific asset."""
    store = _get_store()
    try:
        result = store.get_evaluation(asset_id)
        if result is None:
            raise HTTPException(status_code=404, detail="No evaluation found")
        return result
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Assets — reconciliation for asset-tracker worker
# ---------------------------------------------------------------------------


@app.get("/assets/{asset_id}")
def get_asset(asset_id: str):
    """Get asset info by asset_id (ETag)."""
    store = _get_store()
    try:
        result = store.get_asset(asset_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return result
    finally:
        store.close()


@app.post("/assets/reconcile", response_model=ReconcileResponse)
def reconcile_asset(request: ReconcileRequest):
    """Update asset path after a file move. Called by asset-tracker worker.

    Uses ETag as asset_id to match the moved file to its existing record.
    """
    store = _get_store()
    try:
        found = store.reconcile_asset(request.asset_id, request.new_path)
        return ReconcileResponse(
            found=found,
            asset_id=request.asset_id,
            new_path=request.new_path,
        )
    finally:
        store.close()


@app.post("/assets/stale")
def mark_stale(request: StaleRequest):
    """Mark an asset as stale (file deleted from R2). Called by asset-tracker."""
    store = _get_store()
    try:
        store.mark_asset_stale(request.path)
        return {"status": "ok", "path": request.path}
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Remove / Reset
# ---------------------------------------------------------------------------


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
    """Delete all indexed data, assets, and evaluations."""
    store = _get_store()
    try:
        s = store.get_stats()
        for f in s["source_files"]:
            store.remove_file(f)
        # Also clear assets and evaluations
        with store._conn.cursor() as cur:
            cur.execute("DELETE FROM evaluations")
            cur.execute("DELETE FROM assets")
        store._conn.commit()
        return {
            "removed_chunks": s["total_chunks"],
            "removed_files": s["unique_source_files"],
        }
    finally:
        store.close()
