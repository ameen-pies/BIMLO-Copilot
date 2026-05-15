import re
import json
import ast
from typing import Optional


def _parse_llm_json(raw: str) -> Optional[dict]:
    """Robust JSON parsing (same logic as in main.py intent/report)."""
    if not raw:
        return None
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\{[\s\S]*?\}', clean)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    try:
        result = ast.literal_eval(clean)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    return None


def test_parse_standard_json():
    assert _parse_llm_json('{"wants_report": true, "mentioned_files": []}') == {"wants_report": True, "mentioned_files": []}


def test_parse_with_fences():
    assert _parse_llm_json('```json\n{"wants_report": false}\n```') == {"wants_report": False}


def test_parse_with_preamble():
    result = _parse_llm_json("Here is the JSON:\n{\"wants_report\": true}")
    assert result is not None and result.get("wants_report") is True


def test_parse_single_quotes():
    result = _parse_llm_json("{'wants_report': False, 'mentioned_files': ['doc.pdf']}")
    assert result is not None


def test_report_regex():
    REPORT_RE = re.compile(
        r'\b(?:'
        r'(?:make|create|generate|write|build|produce|prepare|draft|do|give\s+me|get\s+me)'
        r'\s+(?:me\s+)?(?:a\s+|an\s+)?(?:full\s+|detailed\s+|brief\s+)?report'
        r'|report\s+(?:on|about|for|regarding|from)'
        r'|(?:make|create|generate|write|do)\s+(?:me\s+)?(?:a\s+)?(?:summary\s+)?document'
        r'|generate\s+(?:a\s+)?(?:pdf|word\s+doc|summary\s+report)'
        r'|download\s+(?:a\s+)?report'
        r'|rapport\s+sur'
        r'|fais?\s+(?:moi\s+)?(?:un\s+)?rapport'
        r'|cr[eé]e?\s+(?:un\s+)?rapport'
        r'|g[eé]n[eè]re?\s+(?:un\s+)?rapport'
        r')',
        re.IGNORECASE,
    )
    assert REPORT_RE.search("generate a report on 5G")
    assert REPORT_RE.search("Make me a report about antennas")
    assert REPORT_RE.search("rapport sur le réseau")
    assert REPORT_RE.search("crée un rapport")
    assert not REPORT_RE.search("what is 5G?")
    assert not REPORT_RE.search("summarise this document")
