"""FastAPI web API for SentrySearch."""

import os
import shutil
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(
    title="SentrySearch",
    description="Semantic search over video footage via natural language queries.",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    r2_keys: list[str]
    chunk_duration: int = 30
    overlap: int = 5
    preprocess: bool = True
    target_resolution: int = 480
    target_fps: int = 5
    skip_still: bool = True


class IndexResponse(BaseModel):
    new_chunks: int
    new_files: int
    skipped_chunks: int
    total_chunks: int
    total_files: int


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    threshold: float = 0.41
    trim: bool = False
    save_top: int | None = None


class SearchResult(BaseModel):
    source_file: str
    start_time: float
    end_time: float
    similarity_score: float
    clip_url: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


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
    """Index video files from R2 for searching."""
    from .chunker import chunk_video, is_still_frame_chunk, preprocess_chunk
    from .embedder import get_embedder, reset_embedder

    store = _get_store()
    r2 = _get_r2()

    try:
        embedder = get_embedder("gemini")
        new_files = 0
        new_chunks = 0
        skipped_chunks = 0

        for r2_key in request.r2_keys:
            if store.is_indexed(r2_key):
                continue

            local_path = r2.download_temp(r2_key)
            tmp_dir = None

            try:
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
                    if request.skip_still and is_still_frame_chunk(
                        chunk["chunk_path"]
                    ):
                        skipped_chunks += 1
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
                    new_files += 1
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
            new_chunks=new_chunks,
            new_files=new_files,
            skipped_chunks=skipped_chunks,
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
    """Search indexed footage with a natural language query."""
    from .embedder import get_embedder, reset_embedder
    from .search import search_footage as _search
    from .trimmer import _safe_filename, trim_clip

    store = _get_store()

    try:
        if store.get_stats()["total_chunks"] == 0:
            raise HTTPException(
                status_code=404,
                detail="No indexed footage. POST /index first.",
            )

        get_embedder("gemini")

        n = request.n_results
        if request.save_top and request.save_top > n:
            n = request.save_top

        results = _search(request.query, store, n_results=n)

        if not results:
            return SearchResponse(results=[])

        trim_count = request.save_top or (1 if request.trim else 0)
        r2 = _get_r2() if trim_count > 0 else None
        output = []

        for i, r in enumerate(results):
            clip_url = None

            if i < trim_count and r["similarity_score"] >= request.threshold:
                try:
                    local_video = r2.download_temp(r["source_file"])
                    clip_name = _safe_filename(
                        r["source_file"], r["start_time"], r["end_time"]
                    )
                    clip_dir = tempfile.mkdtemp(prefix="sentrysearch_clips_")
                    clip_path = os.path.join(clip_dir, clip_name)

                    trim_clip(
                        source_file=local_video,
                        start_time=r["start_time"],
                        end_time=r["end_time"],
                        output_path=clip_path,
                    )

                    clip_key = f"clips/{clip_name}"
                    r2.upload(clip_path, clip_key)
                    clip_url = r2.presigned_url(clip_key)

                    os.unlink(local_video)
                    shutil.rmtree(clip_dir, ignore_errors=True)
                except Exception:
                    pass  # Trimming is best-effort

            output.append(
                SearchResult(
                    source_file=r["source_file"],
                    start_time=r["start_time"],
                    end_time=r["end_time"],
                    similarity_score=r["similarity_score"],
                    clip_url=clip_url,
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
