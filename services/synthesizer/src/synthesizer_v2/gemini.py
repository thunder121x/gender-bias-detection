"""
gemini.py — Native Google AI Studio REST client for synthesizer_v2
===================================================================
Calls https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
exactly the same way the original synthesize_binary.py does.
"""

import json
import ssl
from typing import Any, Dict, Optional
from urllib import error, request


def _make_ssl_context() -> ssl.SSLContext:
    """
    Return an SSL context with verified CA certs.
    On macOS with a stock Python 3.x install the default context may not find
    the system CA bundle.  We try certifi first, then fall back to the default
    context (which works on most Linux/CI environments already).
    """
    try:
        import certifi  # type: ignore
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        pass
    # certifi not installed — let Python find the CAs itself
    return ssl.create_default_context()


def create_chat_completion(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    response_schema: Optional[Dict[str, Any]] = None,
    timeout: int = 180,
) -> str:
    """
    Call the Gemini native generateContent endpoint.
    Returns the raw text content of the first candidate.

    If ``response_schema`` is provided it is passed as ``responseSchema``
    inside ``generationConfig`` — this is Gemini's structured-output feature
    that enforces the exact JSON shape (including enum values) at the sampler
    level, eliminating wrong-subtype outputs without any prompt-level JSON
    instructions.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={api_key}"
    )
    gen_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "responseMimeType": "application/json",
    }
    if response_schema is not None:
        gen_config["responseSchema"] = response_schema

    body: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": gen_config,
    }
    encoded = json.dumps(body).encode("utf-8")
    req = request.Request(
        url=url,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    ssl_ctx = _make_ssl_context()
    https_handler = request.HTTPSHandler(context=ssl_ctx)
    opener = request.build_opener(https_handler)
    try:
        with opener.open(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc
    except (TimeoutError, OSError) as exc:
        raise RuntimeError(f"Timeout/IO error: {exc}") from exc

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc
