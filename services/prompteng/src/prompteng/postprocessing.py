import json
from typing import Optional

def clean_llm_output(text):
    if not isinstance(text, str):
        return text

    text = text.replace("```json", "").replace("```", "")

    return (
        text.replace("\n", " ")
            .replace("|", "")
            .replace("،", ",")
            .strip()
    )


def fix_span_format(span: Optional[list]):
    """
    Normalize span into [start, end] format.

    Handles:
      5 → [5, 5]
      "5" → [5, 5]
      [5] → [5, 5]
      [5, 7] → [5, 7]
      [[4]] → [4, 4]
      [[5,6]] → [5, 6]
      [["8"]] → [8, 8]
    """
    if isinstance(span, list) and len(span) == 2:
        return span

    try:
        # Case 1: integer
        if isinstance(span, int):
            return [span, span]

        # Case 2: single string integer
        if isinstance(span, str) and span.isdigit():
            x = int(span)
            return [x, x]

        # Case 3: list handling (flat or nested)
        if isinstance(span, list):
            # Nested list e.g. [[4]], [["8"]], [[3,5]]
            if len(span) == 1 and isinstance(span[0], list):
                inner = span[0]
                return fix_span_format(inner)

            # Flat list e.g. [4], ["4"], [4,7]
            if len(span) == 1:
                x = span[0]
                if isinstance(x, str) and x.isdigit():
                    return [int(x), int(x)]
                if isinstance(x, int):
                    return [x, x]

            if len(span) >= 2:
                # Map to integers when possible
                s, e = span[0], span[1]

                # Convert string "5" to int
                if isinstance(s, str) and s.isdigit():
                    s = int(s)
                if isinstance(e, str) and e.isdigit():
                    e = int(e)

                return [s, e]

        # Unhandled formats → skip
        return None

    except Exception:
        return None


def postprocess_json_output(text):
    """
    Full postprocessing:
    - cleanup text
    - json parse
    - fix rationales & triggers
    """

    cleaned = clean_llm_output(text)

    # Try JSON parsing
    try:
        data = json.loads(cleaned)
    except:
        # return cleaned text if JSON invalid
        return {"error": "invalid_json", "raw": cleaned}

    # ---- Fix rationale spans ----
    if "rationales" in data:
        fixed = []

        if isinstance(data["rationales"], list):
            for x in data["rationales"]:
                span = fix_span_format(x)
                if span:
                    fixed.append(span)
        elif isinstance(data["rationales"], (int, str)):
            span = fix_span_format(data["rationales"])
            if span:
                fixed.append(span)
        else:
            fixed = []

        data["rationales"] = fixed

    # ---- Fix triggers the same way ----
    if "triggers" in data:
        fixed = []

        if isinstance(data["triggers"], list):
            for x in data["triggers"]:
                span = fix_span_format(x)
                if span:
                    fixed.append(span)
        elif isinstance(data["triggers"], (int, str)):
            span = fix_span_format(data["triggers"])
            if span:
                fixed.append(span)
        else:
            fixed = []

        data["triggers"] = fixed

    return data