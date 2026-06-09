import json
from typing import Any, Dict
from urllib import error, request


def create_chat_completion(
    *, base_url: str, api_key: str, payload: Dict[str, Any], timeout: int = 180
) -> Dict[str, Any]:
    """Call OpenRouter (or any OpenAI-compatible) chat completions endpoint."""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {msg}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc
