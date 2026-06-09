"""
gemini_batch.py — Vertex AI Gemini Batch Prediction support for synthesizer_v2
================================================================================
Provides async batch inference via Google Cloud Vertex AI batchPredictionJobs
(50% cost discount vs real-time, up to 200 000 requests/job).

Flow:
  1. build_batch_jsonl()   → builds JSONL string (Gemini native API format)
  2. upload_to_gcs()       → uploads input JSONL to GCS
  3. submit_batch_job()    → creates Vertex AI batchPredictionJob, returns job name
  4. poll_job()            → polls until SUCCEEDED / FAILED / CANCELLED
  5. download_and_parse_output() → downloads output JSONL from GCS, validates items
  6. recover_partial()     → like download_and_parse_output but tolerates partial jobs

Auth:
  Uses google-auth library (ADC / GOOGLE_APPLICATION_CREDENTIALS) when available,
  or falls back to `gcloud auth print-access-token` via subprocess.

References:
  https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
  https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.batchPredictionJobs
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Token acquisition
# ---------------------------------------------------------------------------


def _get_access_token() -> str:
    """Return a Bearer token using google-auth ADC, or gcloud fallback."""
    try:
        import google.auth  # type: ignore
        import google.auth.transport.requests  # type: ignore

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
        token: str = credentials.token
        return token
    except Exception:
        pass  # fall through to gcloud

    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:
        raise RuntimeError(
            "Could not obtain GCP access token. "
            "Install google-auth (`pip install google-auth`) or run `gcloud auth login`."
        ) from exc


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only, plus GCS / Vertex REST)
# ---------------------------------------------------------------------------


def _http_request(
    url: str,
    method: str = "GET",
    body: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    token: Optional[str] = None,
) -> Tuple[int, str]:
    """Thin wrapper around urllib; returns (status_code, body_text)."""
    hdrs: Dict[str, str] = headers or {}
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


# ---------------------------------------------------------------------------
# Build JSONL
# ---------------------------------------------------------------------------


def build_batch_jsonl(
    *,
    mode: str,
    total: int,
    prompts: Dict[str, Dict[str, str]],
    seeds: Dict[str, List[str]],
    model: str,
    temperature: float = 0.95,
    max_tokens: int = 8192,
    batch_size: int = 25,
    seed_sample_size: int = 8,
) -> str:
    """
    Build a JSONL string where each line is one Gemini native API request.

    Each request asks for `batch_size` items; the total number of requests
    is ceil(total / batch_size).

    The Gemini native format inside each JSONL row:
    {
      "request": {
        "contents": [{"role": "user", "parts": [{"text": "..."}]}],
        "systemInstruction": {"parts": [{"text": "..."}]},
        "generationConfig": {
          "temperature": ...,
          "maxOutputTokens": ...,
          "responseMimeType": "application/json"
        }
      }
    }
    """
    import math

    n_requests = math.ceil(total / batch_size)
    prompt_config = prompts[mode]
    seed_pool = seeds[mode]

    lines: List[str] = []
    for req_idx in range(n_requests):
        # Last request might need fewer items
        this_count = batch_size if req_idx < n_requests - 1 else (total - batch_size * (n_requests - 1))

        k = min(seed_sample_size, len(seed_pool))
        sampled_seeds = random.sample(seed_pool, k)
        seed_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sampled_seeds))

        user_text = prompt_config["user"].replace("{{COUNT}}", str(this_count)).replace("{{SEED_EXAMPLES}}", seed_text)
        system_text = prompt_config["system"]

        request_obj: Dict[str, Any] = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_text}],
                    }
                ],
                "systemInstruction": {"parts": [{"text": system_text}]},
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                },
            }
        }
        lines.append(json.dumps(request_obj, ensure_ascii=False))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GCS upload / download
# ---------------------------------------------------------------------------


def upload_to_gcs(
    *,
    content: str,
    bucket: str,
    blob_name: str,
    token: Optional[str] = None,
) -> str:
    """
    Upload `content` (UTF-8 text) to gs://<bucket>/<blob_name>.

    Tries google-cloud-storage first; falls back to GCS XML API via urllib.

    Returns the GCS URI: gs://<bucket>/<blob_name>
    """
    if token is None:
        token = _get_access_token()

    # Try google-cloud-storage library first
    try:
        from google.cloud import storage as gcs  # type: ignore

        client = gcs.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        blob.upload_from_string(content.encode("utf-8"), content_type="application/jsonl")
        return f"gs://{bucket}/{blob_name}"
    except ImportError:
        pass  # fall through to REST

    # GCS JSON API upload
    encoded = content.encode("utf-8")
    encoded_blob = urllib.parse.quote(blob_name, safe="")
    upload_url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/"
        f"{urllib.parse.quote(bucket, safe='')}/o"
        f"?uploadType=media&name={encoded_blob}"
    )
    status, body = _http_request(
        upload_url,
        method="POST",
        body=encoded,
        headers={"Content-Type": "application/jsonl"},
        token=token,
    )
    if status not in (200, 201):
        raise RuntimeError(f"GCS upload failed [{status}]: {body}")
    return f"gs://{bucket}/{blob_name}"


def list_gcs_blobs(
    *,
    bucket: str,
    prefix: str,
    token: Optional[str] = None,
) -> List[str]:
    """Return a list of gs:// URIs matching bucket + prefix."""
    if token is None:
        token = _get_access_token()

    encoded_prefix = urllib.parse.quote(prefix, safe="")
    url = f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o?prefix={encoded_prefix}"
    status, body = _http_request(url, token=token)
    if status != 200:
        raise RuntimeError(f"GCS list failed [{status}]: {body}")
    data = json.loads(body)
    items = data.get("items", [])
    return [f"gs://{bucket}/{item['name']}" for item in items]


def download_from_gcs(
    *,
    gcs_uri: str,
    token: Optional[str] = None,
) -> str:
    """Download text content from a gs:// URI."""
    if token is None:
        token = _get_access_token()

    # Parse gs://bucket/blob
    without_scheme = gcs_uri[len("gs://") :]
    bucket, _, blob_name = without_scheme.partition("/")

    # Try google-cloud-storage first
    try:
        from google.cloud import storage as gcs_lib  # type: ignore

        client = gcs_lib.Client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_name)
        return blob.download_as_text(encoding="utf-8")
    except ImportError:
        pass

    encoded_blob = urllib.parse.quote(blob_name, safe="")
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o/{encoded_blob}?alt=media"
    )
    status, body = _http_request(url, token=token)
    if status != 200:
        raise RuntimeError(f"GCS download failed [{status}]: {body}")
    return body


# ---------------------------------------------------------------------------
# Vertex AI Batch Prediction Job
# ---------------------------------------------------------------------------


def submit_batch_job(
    *,
    input_gcs_uri: str,
    output_gcs_prefix: str,
    model: str,
    project: str,
    location: str = "us-central1",
    job_display_name: str = "synthesizer-v2-batch",
    token: Optional[str] = None,
) -> str:
    """
    Submit a Vertex AI batchPredictionJob.

    model should be the publisher model resource name, e.g.:
      "publishers/google/models/gemini-2.5-flash-001"
    or a full model path:
      "projects/{project}/locations/{location}/publishers/google/models/gemini-2.5-flash-001"

    Returns the job resource name, e.g.:
      "projects/{project}/locations/{location}/batchPredictionJobs/{id}"
    """
    if token is None:
        token = _get_access_token()

    # Normalize model name to publisher format
    if not model.startswith("publishers/") and not model.startswith("projects/"):
        # Strip provider prefix if present (e.g. "google/gemini-2.5-flash-001")
        base_model = model.split("/")[-1] if "/" in model else model
        model_resource = f"publishers/google/models/{base_model}"
    else:
        model_resource = model

    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/batchPredictionJobs"

    payload: Dict[str, Any] = {
        "displayName": job_display_name,
        "model": model_resource,
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [input_gcs_uri]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": output_gcs_prefix},
        },
    }

    body = json.dumps(payload).encode("utf-8")
    status, resp_body = _http_request(
        url,
        method="POST",
        body=body,
        headers={"Content-Type": "application/json"},
        token=token,
    )
    if status not in (200, 201):
        raise RuntimeError(f"Batch job submission failed [{status}]: {resp_body}")

    data = json.loads(resp_body)
    job_name: str = data["name"]
    print(f"  [gemini-batch] job submitted: {job_name}", file=sys.stderr)
    return job_name


def poll_job(
    *,
    job_name: str,
    location: str = "us-central1",
    poll_interval: int = 60,
    timeout_hours: float = 25.0,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Poll batchPredictionJob until terminal state (SUCCEEDED/FAILED/CANCELLED/JOB_STATE_CANCELLED).

    Returns the final job resource dict.

    Terminal states:
      JOB_STATE_SUCCEEDED, JOB_STATE_FAILED, JOB_STATE_CANCELLED,
      JOB_STATE_EXPIRED, JOB_STATE_PARTIALLY_SUCCEEDED
    """
    if token is None:
        token = _get_access_token()

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
    }

    url = f"https://{location}-aiplatform.googleapis.com/v1/{job_name}"
    deadline = time.time() + timeout_hours * 3600
    start = time.time()

    state = ""
    while time.time() < deadline:
        status, body = _http_request(url, token=token)
        if status != 200:
            print(f"  [gemini-batch] poll error [{status}]: {body}", file=sys.stderr)
            time.sleep(poll_interval)
            # Refresh token on auth errors
            if status in (401, 403):
                token = _get_access_token()
            continue

        data = json.loads(body)
        state = data.get("state", "")
        elapsed = (time.time() - start) / 60

        print(
            f"  [gemini-batch] state={state} elapsed={elapsed:.1f}m",
            file=sys.stderr,
        )

        if state in terminal_states:
            return data

        time.sleep(poll_interval)

    raise TimeoutError(f"Batch job did not complete within {timeout_hours}h. Last state: {state}. Job: {job_name}")


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _parse_output_jsonl(
    jsonl_text: str,
    mode: str,
    expected_label: str,
    expected_subtype: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Parse the output JSONL from a Vertex AI batch job.

    Each line has the shape:
    {
      "status": "",
      "request": {...},
      "response": {
        "candidates": [{
          "content": {"parts": [{"text": "..."}]}
        }]
      }
    }

    Returns (valid_items, n_rows, n_errors).
    """
    from synthesizer_v2.generate import _parse_response, _validate_item  # local import

    valid_items: List[Dict[str, Any]] = []
    n_rows = 0
    n_errors = 0

    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        n_rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            n_errors += 1
            continue

        row_status = row.get("status", "")
        if row_status and row_status != "OK" and row_status != "":
            # Non-empty status usually means the individual request failed
            error_msg = row_status if isinstance(row_status, str) else json.dumps(row_status)
            print(f"  [gemini-batch] row error: {error_msg}", file=sys.stderr)
            n_errors += 1
            continue

        response = row.get("response", {})
        candidates = response.get("candidates", [])
        if not candidates:
            n_errors += 1
            continue

        try:
            raw_text: str = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            n_errors += 1
            continue

        items = _parse_response(raw_text)
        for item in items:
            if _validate_item(item, mode):
                valid_items.append(item)

    return valid_items, n_rows, n_errors


def download_and_parse_output(
    *,
    output_gcs_prefix: str,
    mode: str,
    expected_label: str,
    expected_subtype: str,
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Download all output JSONL files from GCS prefix and parse them.

    Vertex AI writes output to:
      <output_gcs_prefix>/<job-id>/predictions_*.jsonl

    Returns deduplicated valid items.
    """
    if token is None:
        token = _get_access_token()

    # Parse bucket + prefix
    without_scheme = output_gcs_prefix[len("gs://") :]
    bucket, _, prefix = without_scheme.partition("/")

    blobs = list_gcs_blobs(bucket=bucket, prefix=prefix, token=token)

    # Filter to prediction output files only
    pred_blobs = [b for b in blobs if b.endswith(".jsonl") or "predictions" in b]
    if not pred_blobs:
        # Also accept any .jsonl in the output prefix
        pred_blobs = [b for b in blobs if b.endswith(".jsonl")]

    if not pred_blobs:
        print(
            f"  [gemini-batch] warning: no output JSONL files found under {output_gcs_prefix}",
            file=sys.stderr,
        )
        return []

    all_items: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()
    total_rows = 0
    total_errors = 0

    for blob_uri in pred_blobs:
        print(f"  [gemini-batch] downloading {blob_uri}", file=sys.stderr)
        text = download_from_gcs(gcs_uri=blob_uri, token=token)
        items, n_rows, n_errors = _parse_output_jsonl(text, mode, expected_label, expected_subtype)
        total_rows += n_rows
        total_errors += n_errors

        for item in items:
            t = item.get("text", "")
            if t not in seen_texts:
                seen_texts.add(t)
                all_items.append(item)

    print(
        f"  [gemini-batch] parsed {len(all_items)} valid items from {total_rows} rows ({total_errors} row errors)",
        file=sys.stderr,
    )
    return all_items


def recover_partial(
    *,
    job_name: str,
    output_gcs_prefix: str,
    mode: str,
    expected_label: str,
    expected_subtype: str,
    location: str = "us-central1",
    token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Attempt to recover whatever completed items are available, even if the job
    was cancelled or partially failed.  The output GCS prefix may contain
    partial results written before the job was terminated.
    """
    if token is None:
        token = _get_access_token()

    print(
        f"  [gemini-batch] attempting partial recovery from {output_gcs_prefix}",
        file=sys.stderr,
    )
    return download_and_parse_output(
        output_gcs_prefix=output_gcs_prefix,
        mode=mode,
        expected_label=expected_label,
        expected_subtype=expected_subtype,
        token=token,
    )


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------


def run_batch_pipeline(
    *,
    mode: str,
    total: int,
    model: str,
    project: str,
    location: str,
    gcs_bucket: str,
    gcs_prefix: str,
    prompts: Dict[str, Dict[str, str]],
    seeds: Dict[str, List[str]],
    expected_label: str,
    expected_subtype: str,
    temperature: float = 0.95,
    max_tokens: int = 8192,
    batch_size: int = 25,
    seed_sample_size: int = 8,
    poll_interval: int = 60,
    out_path: Optional[Path] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    Full pipeline: build JSONL → upload → submit job → poll → download → parse.

    Returns validated items (may be fewer than `total` if the model
    didn't produce enough valid items).
    """
    import datetime

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    input_blob = f"{gcs_prefix}/input/{mode}_{timestamp}.jsonl"
    output_prefix = f"gs://{gcs_bucket}/{gcs_prefix}/output/{mode}_{timestamp}"
    input_gcs_uri = f"gs://{gcs_bucket}/{input_blob}"

    # 1. Build JSONL
    print(
        f"  [gemini-batch] building {total} items in batches of {batch_size} …",
        file=sys.stderr,
    )
    jsonl_content = build_batch_jsonl(
        mode=mode,
        total=total,
        prompts=prompts,
        seeds=seeds,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        batch_size=batch_size,
        seed_sample_size=seed_sample_size,
    )
    n_requests = jsonl_content.count("\n") + 1
    print(f"  [gemini-batch] {n_requests} requests in input JSONL", file=sys.stderr)

    if dry_run:
        print("\n=== BATCH INPUT JSONL (first request) ===")
        first_line = jsonl_content.split("\n")[0]
        first_obj = json.loads(first_line)
        print(json.dumps(first_obj, ensure_ascii=False, indent=2))
        print(f"\n[dry-run] would upload to: {input_gcs_uri}")
        print(f"[dry-run] output prefix:    {output_prefix}")
        return []

    # 2. Upload to GCS
    print(f"  [gemini-batch] uploading input to {input_gcs_uri} …", file=sys.stderr)
    token = _get_access_token()
    upload_to_gcs(
        content=jsonl_content,
        bucket=gcs_bucket,
        blob_name=input_blob,
        token=token,
    )

    # 3. Submit job
    job_name = submit_batch_job(
        input_gcs_uri=input_gcs_uri,
        output_gcs_prefix=output_prefix,
        model=model,
        project=project,
        location=location,
        job_display_name=f"synth-v2-{mode}-{timestamp}",
        token=token,
    )

    # Persist job metadata alongside the output file for recovery
    if out_path is not None:
        meta_path = out_path.with_suffix(".batch_job.json")
        meta = {
            "job_name": job_name,
            "input_gcs_uri": input_gcs_uri,
            "output_gcs_prefix": output_prefix,
            "mode": mode,
            "total": total,
            "timestamp": timestamp,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"  [gemini-batch] job metadata saved → {meta_path}", file=sys.stderr)

    # 4. Poll
    print(f"  [gemini-batch] polling every {poll_interval}s …", file=sys.stderr)
    final_job = poll_job(
        job_name=job_name,
        location=location,
        poll_interval=poll_interval,
        token=token,
    )
    final_state = final_job.get("state", "")
    print(f"  [gemini-batch] final state: {final_state}", file=sys.stderr)

    # 5. Download + parse (even on partial success)
    if final_state in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        items = download_and_parse_output(
            output_gcs_prefix=output_prefix,
            mode=mode,
            expected_label=expected_label,
            expected_subtype=expected_subtype,
            token=token,
        )
    elif final_state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
        print(
            f"  [gemini-batch] job ended with {final_state}, attempting partial recovery …",
            file=sys.stderr,
        )
        items = recover_partial(
            job_name=job_name,
            output_gcs_prefix=output_prefix,
            mode=mode,
            expected_label=expected_label,
            expected_subtype=expected_subtype,
            location=location,
            token=token,
        )
    else:
        print(
            f"  [gemini-batch] unexpected state {final_state}, attempting recovery …",
            file=sys.stderr,
        )
        items = recover_partial(
            job_name=job_name,
            output_gcs_prefix=output_prefix,
            mode=mode,
            expected_label=expected_label,
            expected_subtype=expected_subtype,
            location=location,
            token=token,
        )

    print(
        f"  [gemini-batch] collected {len(items)} valid items (requested {total})",
        file=sys.stderr,
    )
    return items
