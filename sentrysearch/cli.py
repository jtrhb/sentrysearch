"""Click-based CLI entry point."""

import os
import platform
import shutil
import subprocess

import click
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", ".env")
MAX_SINGLE_EMBED_DURATION = 120

load_dotenv(_ENV_PATH)
load_dotenv()


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _open_file(path: str) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _handle_error(e: Exception) -> None:
    from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError

    if isinstance(e, GeminiAPIKeyError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, GeminiQuotaError):
        click.secho("Error: " + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, PermissionError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, FileNotFoundError):
        click.secho("Error: " + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, RuntimeError) and "ffmpeg not found" in str(e).lower():
        click.secho(
            "Error: ffmpeg is not available.\n\n"
            "Install it with one of:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  pip fallback:   uv add imageio-ffmpeg",
            fg="red", err=True,
        )
        raise SystemExit(1)
    raise e


@click.group()
def cli():
    """Search and evaluate video footage using natural language queries."""


# -----------------------------------------------------------------------
# init
# -----------------------------------------------------------------------

@cli.command()
def init():
    """Set up your Gemini API key."""
    env_path = _ENV_PATH
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    if os.path.exists(env_path):
        with open(env_path) as f:
            contents = f.read()
        if "GEMINI_API_KEY=" in contents:
            if not click.confirm("API key already configured. Overwrite?", default=False):
                return

    api_key = click.prompt(
        "Enter your Gemini API key\n"
        "  Get one at https://aistudio.google.com/apikey\n"
        "  (input is hidden)",
        hide_input=True,
    )

    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            found = False
            for line in lines:
                if line.startswith("GEMINI_API_KEY="):
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"GEMINI_API_KEY={api_key}\n")
    else:
        with open(env_path, "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")

    os.environ["GEMINI_API_KEY"] = api_key
    click.echo("Validating API key...")
    try:
        from .embedder import get_embedder
        embedder = get_embedder("gemini")
        vec = embedder.embed_query("test")
        if len(vec) != 3072:
            click.secho(f"Unexpected dimension: {len(vec)} (expected 3072).", fg="yellow", err=True)
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"Validation failed: {e}", fg="red", err=True)
        raise SystemExit(1)

    click.secho("Setup complete.", fg="green")


# -----------------------------------------------------------------------
# index
# -----------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--preprocess/--no-preprocess", default=True, show_default=True)
@click.option("--target-resolution", default=480, show_default=True)
@click.option("--target-fps", default=5, show_default=True)
@click.option("--chunk-duration", default=30, show_default=True, help="Only for videos >120s.")
@click.option("--overlap", default=5, show_default=True, help="Only for videos >120s.")
@click.option("--verbose", is_flag=True)
def index(directory, preprocess, target_resolution, target_fps, chunk_duration, overlap, verbose):
    """Index video files for searching. Videos ≤120s embed whole."""
    from .chunker import (
        SUPPORTED_VIDEO_EXTENSIONS, _get_video_duration, chunk_video,
        is_still_frame_chunk, preprocess_chunk, scan_directory,
    )
    from .embedder import get_embedder, reset_embedder
    from .store import SentryStore

    try:
        embedder = get_embedder("gemini")
        videos = [os.path.abspath(directory)] if os.path.isfile(directory) else scan_directory(directory)

        if not videos:
            click.echo(f"No supported video files found ({', '.join(SUPPORTED_VIDEO_EXTENSIONS)}).")
            return

        store = SentryStore()
        total_files = len(videos)
        new_files = 0
        new_chunks = 0

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            if store.is_indexed(abs_path):
                click.echo(f"Skipping ({file_idx}/{total_files}): {basename} (already indexed)")
                continue

            duration = _get_video_duration(abs_path)

            if duration <= MAX_SINGLE_EMBED_DURATION:
                click.echo(f"Indexing ({file_idx}/{total_files}): {basename} ({duration:.0f}s, whole)")
                embed_path = abs_path
                if preprocess:
                    embed_path = preprocess_chunk(abs_path, target_resolution=target_resolution, target_fps=target_fps)

                embedding = embedder.embed_video_chunk(embed_path, verbose=verbose)
                store.add_chunks([{
                    "source_file": abs_path,
                    "start_time": 0.0,
                    "end_time": duration,
                    "embedding": embedding,
                }])
                new_files += 1
                new_chunks += 1

                if embed_path != abs_path:
                    try:
                        os.unlink(embed_path)
                    except OSError:
                        pass
            else:
                chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
                num_chunks = len(chunks)
                click.echo(f"Indexing ({file_idx}/{total_files}): {basename} ({duration:.0f}s, {num_chunks} chunks)")
                embedded = []
                files_to_cleanup = []

                for chunk_idx, chunk in enumerate(chunks, 1):
                    if is_still_frame_chunk(chunk["chunk_path"], verbose=verbose):
                        files_to_cleanup.append(chunk["chunk_path"])
                        continue
                    click.echo(f"  Chunk {chunk_idx}/{num_chunks}")
                    embed_path = chunk["chunk_path"]
                    if preprocess:
                        embed_path = preprocess_chunk(embed_path, target_resolution=target_resolution, target_fps=target_fps)
                        if embed_path != chunk["chunk_path"]:
                            files_to_cleanup.append(embed_path)
                    embedding = embedder.embed_video_chunk(embed_path, verbose=verbose)
                    embedded.append({**chunk, "embedding": embedding})
                    files_to_cleanup.append(chunk["chunk_path"])

                for f in files_to_cleanup:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
                if chunks:
                    shutil.rmtree(os.path.dirname(chunks[0]["chunk_path"]), ignore_errors=True)
                if embedded:
                    store.add_chunks(embedded)
                    new_files += 1
                    new_chunks += len(embedded)

        s = store.get_stats()
        click.echo(f"\nIndexed {new_chunks} embeddings from {new_files} files. "
                    f"Total: {s['total_chunks']} from {s['unique_source_files']} files.")
        store.close()
    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True)
@click.option("--threshold", default=0.41, show_default=True, type=float)
@click.option("--verbose", is_flag=True)
def search(query, n_results, threshold, verbose):
    """Search indexed footage with a natural language QUERY."""
    from .embedder import get_embedder, reset_embedder
    from .search import search_footage
    from .store import SentryStore

    try:
        store = SentryStore()
        if store.get_stats()["total_chunks"] == 0:
            click.echo("No indexed footage. Run `sentrysearch index <dir>` first.")
            store.close()
            return

        get_embedder("gemini")
        results = search_footage(query, store, n_results=n_results, verbose=verbose)

        if not results:
            click.echo("No results found.")
            store.close()
            return

        best = results[0]["similarity_score"]
        if best < threshold:
            click.secho(f"(low confidence — best: {best:.2f})", fg="yellow", err=True)

        for i, r in enumerate(results, 1):
            asset_id = r["source_file"]
            asset = store.get_asset(asset_id)
            display = asset["current_path"] if asset else asset_id
            display = os.path.basename(display)
            score = r["similarity_score"]
            start = _fmt_time(r["start_time"])
            end = _fmt_time(r["end_time"])
            fmt = f"[{score:.6f}]" if verbose else f"[{score:.2f}]"
            click.echo(f"  #{i} {fmt} {display} @ {start}-{end}")

        store.close()
    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# evaluate
# -----------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--no-similarity", is_flag=True, default=False, help="Skip similarity check.")
@click.option("--verbose", is_flag=True, help="Show all issues including nitpicks.")
def evaluate(path, no_similarity, verbose):
    """Evaluate video quality: scoring + defect detection in one pass."""
    from .chunker import SUPPORTED_VIDEO_EXTENSIONS, scan_directory
    from .embedder import reset_embedder
    from .evaluator import VideoEvaluator
    from .store import SentryStore, detect_index

    try:
        evaluator = VideoEvaluator()

        store = None
        if not no_similarity:
            backend, _ = detect_index()
            if backend is not None:
                store = SentryStore()

        videos = [os.path.abspath(path)] if os.path.isfile(path) else scan_directory(path)
        if not videos:
            click.echo(f"No supported video files found ({', '.join(SUPPORTED_VIDEO_EXTENSIONS)}).")
            return

        for i, video_path in enumerate(videos, 1):
            basename = os.path.basename(video_path)
            click.echo(f"\nEvaluating ({i}/{len(videos)}): {basename}")

            result = evaluator.evaluate(video_path, store=store)

            # Grade
            grade = result["quality_grade"]
            grade_score = result["grade_score"]
            overall = result["overall_score"]
            color = "green" if grade in ("A", "B") else "yellow" if grade == "C" else "red"
            click.secho(f"  Grade: {grade} ({grade_score}/100)  Overall: {overall}/100", fg=color, bold=True)

            # Consistency + AI
            con = result["consistency"]
            ai = result["ai_detection"]
            click.echo(f"  Consistency: character {con['character']}, scene {con['scene']} (avg {con['average']})")
            ai_label = 'low AI' if ai['score'] < 40 else 'moderate' if ai['score'] < 70 else 'high AI'
            click.echo(f"  AI score:    {ai['score']}/100 ({ai_label})")

            # Similarity
            sim = result["similarity"]
            if sim["similar_to"]:
                click.echo(f"  Similarity:  {sim['max_similarity']:.2%} to {os.path.basename(sim['similar_to'])}")
            else:
                click.echo("  Similarity:  no indexed assets to compare")

            # Category scores
            cats = [("temporal", "时序"), ("visual", "视觉"), ("character", "角色"),
                    ("audio", "音频"), ("composition", "构图"), ("coherence", "连贯")]
            cat_scores = result["category_scores"]
            parts = []
            for key, label in cats:
                s = cat_scores.get(key)
                parts.append(f"{label}:{s}" if s is not None else f"{label}:N/A")
            click.echo(f"  Categories:  {' | '.join(parts)}")

            # Severity counts
            counts = result["severity_counts"]
            sev_parts = []
            if counts.get("critical", 0):
                sev_parts.append(click.style(f"{counts['critical']} critical", fg="red", bold=True))
            if counts.get("major", 0):
                sev_parts.append(click.style(f"{counts['major']} major", fg="red"))
            if counts.get("minor", 0):
                sev_parts.append(click.style(f"{counts['minor']} minor", fg="yellow"))
            if counts.get("nitpick", 0):
                sev_parts.append(f"{counts['nitpick']} nitpick")
            if sev_parts:
                click.echo(f"  Issues:      {', '.join(sev_parts)}")
            else:
                click.secho("  No issues found!", fg="green")

            # Summary
            if result["summary"]:
                click.echo(f"  {result['summary']}")

            # Issue details
            issues = result["issues"]
            severity_rank = {"critical": 0, "major": 1, "minor": 2, "nitpick": 3}
            min_rank = 3 if verbose else 2
            shown = [iss for iss in issues if severity_rank.get(iss["severity"], 3) <= min_rank]

            if shown:
                click.echo(f"\n  {'Sev':<9} {'Type':<25} {'Time':<12} Description")
                click.echo(f"  {'---':<9} {'----':<25} {'----':<12} -----------")
                for iss in shown:
                    sev = iss["severity"]
                    sev_color = "red" if sev in ("critical", "major") else "yellow" if sev == "minor" else None
                    sev_str = click.style(f"{sev:<9}", fg=sev_color) if sev_color else f"{sev:<9}"
                    click.echo(f"  {sev_str} {iss['type']:<25} {iss['timestamp']:<12} {iss['description']}")

            if store is not None:
                store.save_evaluation(os.path.abspath(video_path), result)

        if store is not None:
            store.close()
    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
def stats():
    """Print index statistics."""
    from .store import SentryStore, detect_index

    backend, _ = detect_index()
    if backend is None:
        click.echo("Index is empty.")
        return

    store = SentryStore()
    s = store.get_stats()
    if s["total_chunks"] == 0:
        click.echo("Index is empty.")
        store.close()
        return

    click.echo(f"Total embeddings:  {s['total_chunks']}")
    click.echo(f"Source files:      {s['unique_source_files']}")
    click.echo(f"Backend:           gemini")
    click.echo("\nIndexed files:")
    for f in s["source_files"]:
        click.echo(f"  {f}")
    store.close()


# -----------------------------------------------------------------------
# reset
# -----------------------------------------------------------------------

@cli.command()
@click.confirmation_option(prompt="Delete all data (index + evaluations + assets)?")
def reset():
    """Delete all indexed data, evaluations, and asset records."""
    from .store import SentryStore, detect_index

    backend, _ = detect_index()
    if backend is None:
        click.echo("Already empty.")
        return

    store = SentryStore()
    s = store.get_stats()
    if s["total_chunks"] == 0:
        click.echo("Already empty.")
        store.close()
        return

    for f in s["source_files"]:
        store.remove_file(f)
    with store._conn.cursor() as cur:
        cur.execute("DELETE FROM evaluations")
        cur.execute("DELETE FROM assets")
    store._conn.commit()

    click.echo(f"Removed {s['total_chunks']} embeddings, assets, and evaluations.")
    store.close()


# -----------------------------------------------------------------------
# remove
# -----------------------------------------------------------------------

@cli.command()
@click.argument("files", nargs=-1, required=True)
def remove(files):
    """Remove specific files from the index."""
    from .store import SentryStore, detect_index

    backend, _ = detect_index()
    if backend is None:
        click.echo("Index is empty.")
        return

    store = SentryStore()
    s = store.get_stats()
    if s["total_chunks"] == 0:
        click.echo("Index is empty.")
        store.close()
        return

    total_removed = 0
    for pattern in files:
        matches = [f for f in s["source_files"] if pattern in f]
        if not matches:
            click.echo(f"No match for '{pattern}'")
            continue
        for source_file in matches:
            removed = store.remove_file(source_file)
            click.echo(f"Removed {removed} embeddings from {source_file}")
            total_removed += removed

    if total_removed:
        click.echo(f"\nTotal: {total_removed} removed.")
    store.close()
