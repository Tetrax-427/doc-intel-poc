"""
batch_runner.py
----------------
Runs "upload -> extract" pipelines for a batch of files, N concurrently
(capped by `max_parallel`), instead of one file at a time.

Design notes:
- Each file's own pipeline (upload then extract) is sequential internally -
  extraction needs the document_id from upload - but different files'
  pipelines run concurrently with each other via an asyncio.Semaphore.
- Every per-file error is caught and turned into a row with an "error"
  field rather than raising, so one bad file never aborts the batch.
- `progress_cb(completed_count, total, filename, ok)` is called after each
  file finishes. Because `run_batch` drives its coroutine via
  `asyncio.run()` on the SAME thread that's executing the Streamlit
  script (see `_run_coro` below), calls to Streamlit widgets from inside
  progress_cb are safe in the common case and update live.
"""
import asyncio

import httpx

from api_client import AsyncDocIntelClient, ApiError


def _run_coro(coro):
    """
    Runs an async coroutine to completion and returns its result.

    Streamlit's script execution does not normally have its own asyncio
    event loop, so `asyncio.run()` works directly and keeps everything on
    the same thread - which matters because Streamlit widget calls (used
    in progress callbacks) only work correctly on the thread running the
    script.

    Defensive fallback: if something in the environment DOES already have
    a running loop (rare, but possible depending on platform/deployment),
    `asyncio.run()` raises RuntimeError. In that case we run the coroutine
    in a dedicated thread with its own fresh loop instead. Streamlit widget
    calls made from inside the coroutine won't render live in that fallback
    path (different thread => no Streamlit ScriptRunContext) - progress
    will just jump to complete at the end rather than updating per file.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "cannot be called from a running event loop" not in str(e):
            raise
        import threading
        box = {}

        def _runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                box["result"] = new_loop.run_until_complete(coro)
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc
            finally:
                new_loop.close()

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        if "error" in box:
            raise box["error"]
        return box["result"]


async def _process_one(async_client, http_client, semaphore, fname, fbytes, fields, schema_name, progress_state, progress_cb):
    async with semaphore:
        try:
            up = await async_client.upload_document(http_client, fbytes, fname)
            doc_id = up.get("document_id")

            result = await async_client.extract_fields(http_client, doc_id, fields, schema_name=schema_name)
            extracted = result.get("extracted", {})

            # Keep extracted RAW/nested here (no flattening) - app.py's
            # render_nested_result / flatten_results_for_export call
            # extract_value() themselves, since a "list" field's value is a
            # list of dicts that a flat row can't hold anyway.
            row = {"filename": fname, "extracted": extracted}
            ok, error_msg = True, None
        except ApiError as e:
            row = {"filename": fname, "extracted": {}, "error": str(e)}
            ok, error_msg = False, f"{fname}: {e}"

    progress_state["completed"] += 1
    if progress_cb:
        progress_cb(progress_state["completed"], progress_state["total"], fname, ok)
    return row, error_msg


async def _run_batch_async(base_url, access_token, timeout, files, fields, schema_name, max_parallel, progress_cb):
    async_client = AsyncDocIntelClient(base_url, access_token, timeout)
    semaphore = asyncio.Semaphore(max_parallel)
    progress_state = {"completed": 0, "total": len(files)}

    limits = httpx.Limits(max_connections=max_parallel, max_keepalive_connections=max_parallel)
    async with httpx.AsyncClient(limits=limits) as http_client:
        tasks = [
            _process_one(async_client, http_client, semaphore, fname, fbytes, fields, schema_name, progress_state, progress_cb)
            for fname, fbytes in files
        ]
        results = await asyncio.gather(*tasks)

    rows = [r for r, _ in results]
    errors = [e for _, e in results if e]
    return rows, errors


def run_batch(base_url: str, access_token: str, timeout: int, files: list, fields: list,
              max_parallel: int, schema_name: str = "extraction_schema", progress_cb=None):
    """
    files: list of (filename, file_bytes)
    fields: schema fields, list of {name, type, description, properties} (nested-capable)
    schema_name: forwarded into the nested_schema wrapper sent to /extract
    progress_cb: optional callable(completed, total, filename, ok)

    Returns (rows, errors) - rows is a list of
    {"filename": ..., "extracted": {...raw nested extracted dict...}, "error"?: str},
    errors is a list of "filename: message" strings for any failed files.
    """
    return _run_coro(_run_batch_async(base_url, access_token, timeout, files, fields, schema_name, max_parallel, progress_cb))