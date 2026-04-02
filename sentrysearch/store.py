"""PostgreSQL + pgvector vector store."""

import hashlib
import os
from datetime import datetime, timezone

import psycopg
from pgvector.psycopg import register_vector


class BackendMismatchError(RuntimeError):
    """Raised on backend mismatch (kept for backward compatibility)."""


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """Deterministic chunk ID from source file + start time."""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_index(database_url: str | None = None) -> tuple[str | None, str | None]:
    """Return ('gemini', None) if indexed data exists, else (None, None)."""
    try:
        url = database_url or os.environ.get("DATABASE_URL", "")
        if not url:
            return None, None
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT EXISTS("
                    "  SELECT 1 FROM information_schema.tables"
                    "  WHERE table_name = 'chunks'"
                    ")"
                )
                if not cur.fetchone()[0]:
                    return None, None
                cur.execute("SELECT EXISTS(SELECT 1 FROM chunks LIMIT 1)")
                return ("gemini", None) if cur.fetchone()[0] else (None, None)
    except Exception:
        return None, None


def detect_backend(database_url: str | None = None) -> str | None:
    """Return the backend that has indexed data, or None."""
    backend, _ = detect_index(database_url)
    return backend


class SentryStore:
    """Persistent vector store backed by PostgreSQL + pgvector."""

    def __init__(self, database_url: str | None = None, **kwargs):
        """Initialize store.

        Args:
            database_url: PostgreSQL connection string. Falls back to
                          DATABASE_URL environment variable.
            **kwargs: Ignored (accepts legacy backend/model/db_path params).
        """
        self._url = database_url or os.environ["DATABASE_URL"]
        self._conn = psycopg.connect(self._url)
        register_vector(self._conn)
        self._ensure_schema()

    def _ensure_schema(self):
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    embedding vector(768) NOT NULL,
                    source_file TEXT NOT NULL,
                    start_time DOUBLE PRECISION NOT NULL,
                    end_time DOUBLE PRECISION NOT NULL,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_source_file_idx
                ON chunks (source_file)
            """)
        self._conn.commit()

    @property
    def collection(self):
        """Not applicable for pgvector — exists for interface compatibility."""
        return None

    def get_backend(self) -> str:
        return "gemini"

    def get_model(self) -> str | None:
        return None

    def check_backend(self, backend: str) -> None:
        if backend != "gemini":
            raise BackendMismatchError(
                f"Only gemini backend is supported, got: {backend}"
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Store a single chunk embedding with metadata."""
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (id, embedding, source_file, start_time, end_time, indexed_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    source_file = EXCLUDED.source_file,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    indexed_at = EXCLUDED.indexed_at
                """,
                (
                    chunk_id, embedding, metadata["source_file"],
                    float(metadata["start_time"]), float(metadata["end_time"]), now,
                ),
            )
        self._conn.commit()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Batch-store chunks. Each dict must have 'embedding' and metadata keys."""
        now = datetime.now(timezone.utc)
        with self._conn.cursor() as cur:
            for chunk in chunks:
                chunk_id = _make_chunk_id(chunk["source_file"], chunk["start_time"])
                cur.execute(
                    """
                    INSERT INTO chunks (id, embedding, source_file, start_time, end_time, indexed_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        indexed_at = EXCLUDED.indexed_at
                    """,
                    (
                        chunk_id, chunk["embedding"], chunk["source_file"],
                        float(chunk["start_time"]), float(chunk["end_time"]), now,
                    ),
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """Return top N results ranked by cosine similarity."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            if cur.fetchone()[0] == 0:
                return []

            cur.execute(
                """
                SELECT source_file, start_time, end_time,
                       1 - distance AS score, distance
                FROM (
                    SELECT source_file, start_time, end_time,
                           embedding <=> %s AS distance
                    FROM chunks
                    ORDER BY distance
                    LIMIT %s
                ) ranked
                """,
                (query_embedding, n_results),
            )
            rows = cur.fetchall()

        return [
            {
                "source_file": row[0],
                "start_time": row[1],
                "end_time": row[2],
                "score": float(row[3]),
                "distance": float(row[4]),
            }
            for row in rows
        ]

    def is_indexed(self, source_file: str) -> bool:
        """Check whether any chunks from source_file are already stored."""
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM chunks WHERE source_file = %s)",
                (source_file,),
            )
            return cur.fetchone()[0]

    def remove_file(self, source_file: str) -> int:
        """Remove all chunks for a given source file. Returns count removed."""
        with self._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE source_file = %s", (source_file,),
            )
            count = cur.rowcount
        self._conn.commit()
        return count

    def get_stats(self) -> dict:
        """Return store statistics."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            total = cur.fetchone()[0]

            if total == 0:
                return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

            cur.execute(
                "SELECT DISTINCT source_file FROM chunks ORDER BY source_file"
            )
            source_files = [row[0] for row in cur.fetchall()]

        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }

    def close(self):
        """Close the database connection."""
        self._conn.close()
