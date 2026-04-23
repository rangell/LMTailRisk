"""Text cleaning and parsing utilities for CRIMSON scoring."""

import json
import os
import re

# ---------------------------------------------------------------------------
# Regex patterns for cleaning radiology report text
# ---------------------------------------------------------------------------

# Collapse spaced-dot abbreviations produced by some models,
# e.g.  ". S. V. C." -> "SVC",  ". P. I. C. C" -> "PICC".
_SPACED_DOT_ABBREV_RE = re.compile(r'\. (?:[A-Z]\. ){1,}[A-Z]\.?')

# Hyphenated compounds split by dots, e.g. ". Port-. A-. Cath" -> "Port-A-Cath".
_HYPHENATED_DOT_RE = re.compile(r'\. (\w+(?:-\. \w+)+)')

# Stray/trailing dots: a lone '.' preceded by whitespace and followed by a
# non-whitespace character — these are tokenization artifacts, not real
# sentence-ending periods (which directly follow a word character).
_STRAY_DOT_RE = re.compile(r'(?<=\s)\. (?=\S)')

# Trailing stray dot(s) at the very end of a string, e.g. "leads. ." -> "leads."
_TRAILING_STRAY_DOT_RE = re.compile(r'\.\s*\.\s*$')

# ---------------------------------------------------------------------------
# Regex for fixing invalid JSON backslash escapes
# ---------------------------------------------------------------------------

# Fix invalid JSON backslash escapes (e.g. \_ or \L) that the model may
# produce when echoing back input text.  Valid JSON escapes
# (\" \\ \/ \b \f \n \r \t \uXXXX) are left untouched.
_INVALID_ESCAPE_RE = re.compile(r'\\(?!["\\\//bfnrtu])')

# ---------------------------------------------------------------------------
# Regex for removing orphan keys in JSON objects
# ---------------------------------------------------------------------------

# Matches a bare string value inside a JSON object — a quoted string preceded
# by a comma and followed by a comma or closing brace, but NOT followed by a
# colon.  Example:  ,"pred_old",  or  ,"pred_old"}
# The leading comma is consumed so removal doesn't leave double commas.
_ORPHAN_KEY_AFTER_COMMA_RE = re.compile(r',\s*"[^"]*"\s*(?=\s*[,}])(?!\s*:)')
# Orphan as the first item in an object:  {"orphan","key":"val"}
_ORPHAN_KEY_AT_START_RE = re.compile(r'(?<=\{)\s*"[^"]*"\s*,\s*(?=")')

# Malformed array-object hybrid:  ["":["R1"]  ->  ["R1"]
# The model starts an array, writes an empty-string "key" with a colon, then
# another array — as if confusing array and object syntax.
_MALFORMED_ARRAY_OBJ_RE = re.compile(r'\[""\s*:\s*\[')

# Missing opening quote on a JSON key:  ,attribute_errors":[  ->  ,"attribute_errors":[
# The model drops the opening " on a key name.  Only fires in structural
# positions: after a JSON value closer (] } " or digit) followed by a comma,
# or directly after an opening brace {.  This avoids false positives on
# commas inside string values like "persists, unchanged".
_MISSING_OPEN_QUOTE_RE = re.compile(
    r'([\]}"0-9]\s*,\s*|{\s*)([a-zA-Z_][a-zA-Z0-9_]*")'
)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _dedup_sentences(text):
    """Remove exact duplicate sentences from repetitive model predictions.

    Some models (CXRMate-RRG24) enter generation loops producing dozens of
    repeated sentences.  This preserves order of first occurrence and rejoins
    with '. '.
    """
    parts = text.split('. ')
    seen = set()
    unique = []
    for p in parts:
        stripped = p.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique.append(p)
    return '. '.join(unique)


def clean_report_text(text):
    """Clean tokenization artifacts in radiology report text.

    Applied transformations (in order):
    1. Strip markdown bold markers (``**``)
    2. Collapse spaced-dot abbreviations ('. S. V. C.' -> 'SVC')
    3. Fix hyphenated compounds split by dots ('. Port-. A-. Cath' -> 'Port-A-Cath')
    4. Remove stray/trailing dots (tokenization artifacts where a lone '.'
       appears before a word, e.g. 'chest . X-ray' -> 'chest X-ray')
    5. Deduplicate exact-match repeated sentences
    """
    if not text:
        return text
    # 1. Markdown bold
    text = text.replace('**', '')
    # 2. Spaced-dot abbreviations
    text = _SPACED_DOT_ABBREV_RE.sub(
        lambda m: ''.join(re.findall(r'[A-Z]', m.group(0))), text
    )
    # 3. Hyphenated compounds
    text = _HYPHENATED_DOT_RE.sub(
        lambda m: m.group(0).replace('-. ', '-').lstrip('. '), text
    )
    # 4. Stray dots
    text = _STRAY_DOT_RE.sub('', text)
    text = _TRAILING_STRAY_DOT_RE.sub('.', text)
    # 5. Dedup
    text = _dedup_sentences(text)
    return text


# ---------------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------------

def _dedupe_keys(pairs):
    """JSON object_pairs_hook that keeps the first occurrence of duplicate keys."""
    result = {}
    for key, value in pairs:
        if key not in result:
            result[key] = value
    return result


def _loads(text):
    """json.loads with duplicate-key handling (keeps first occurrence)."""
    return json.loads(text, object_pairs_hook=_dedupe_keys)


def _is_structural_quote(text, i):
    """Check if the quote at position *i* is likely a JSON structural delimiter.

    Returns True for quotes that delimit JSON keys/values (after ``:``, before
    ``:`` etc.) and False for quotes that are part of prose inside a string
    value (e.g. the model quoting a phrase in an explanation).
    """
    before = text[i - 1] if i > 0 else ''
    after = text[i + 1] if i + 1 < len(text) else ''
    # Quotes right after : [ { are always structural (opening a value/key)
    if before in (':', '[', '{'):
        return True
    # Quotes right before : ] } are always structural (closing a key/value)
    if after in (':', ']', '}'):
        return True
    # For comma-adjacent quotes, check deeper context.
    # Structural ,"  is preceded by a JSON token closer: " ] } or digit
    # Text      ,"  is preceded by a word character (e.g. yesterday,")
    if before == ',':
        pre_comma = text[i - 2] if i >= 2 else ''
        return pre_comma in ('"', ']', '}') or pre_comma.isdigit()
    if after == ',':
        post_comma = text[i + 2] if i + 2 < len(text) else ''
        return post_comma in ('"', '[', '{') or post_comma.isdigit()
    return False


def _fix_unescaped_quotes(text, max_attempts=50):
    """Iteratively escape unescaped double-quotes that break JSON parsing.

    When the model uses literal " inside a JSON string value (e.g. to quote
    a phrase in an explanation), the JSON parser sees it as a string
    terminator.  This function locates each offending quote by parsing,
    catching the error position, searching backwards for a non-structural
    quote, escaping it, and retrying.
    """
    for _ in range(max_attempts):
        try:
            return _loads(text)
        except json.JSONDecodeError as e:
            pos = e.pos
            search_start = max(0, pos - 5)
            found = False
            for i in range(pos, search_start - 1, -1):
                if i < len(text) and text[i] == '"' and (i == 0 or text[i - 1] != '\\'):
                    if _is_structural_quote(text, i):
                        continue
                    text = text[:i] + '\\"' + text[i + 1:]
                    found = True
                    break
            if not found:
                raise
    raise json.JSONDecodeError("Max quote-fix attempts reached", text, 0)


def _fix_orphan_keys(text):
    """Remove bare string values inside JSON objects (orphan keys with no value).

    The model sometimes hallucinates partial key fragments (e.g. ``"pred_old"``)
    that appear as bare values in objects, producing invalid JSON like::

        {"ref_id":"R3","pred_old","pred_id":"P2"}

    This function strips them so the JSON becomes parsable.
    """
    text = _ORPHAN_KEY_AFTER_COMMA_RE.sub('', text)
    text = _ORPHAN_KEY_AT_START_RE.sub('', text)
    return text


def parse_json_response(response, batch_idx=None):
    """Parse model response as JSON, applying progressive fixes.

    Fix pipeline (each step only attempted if prior steps fail):
    1. Raw parse
    2. Remove orphan keys (bare strings in object context)
    3. Fix invalid backslash escapes (``\\L``, ``\\_``, etc.)
    4. Fix unescaped double-quotes inside string values

    Args:
        response: Raw model output string.
        batch_idx: Optional index for error context in batch evaluation.

    Returns:
        Parsed JSON object (dict).

    Raises:
        ValueError: If the response cannot be parsed as JSON.
    """
    # 0. Pre-parse text fixes
    # Escape curly/smart quotes — they always appear inside JSON string values
    response = response.replace('\u201c', '\\"').replace('\u201d', '\\"')
    response = response.replace('\u2018', "\\'").replace('\u2019', "\\'")
    # Fix malformed array-object hybrids:  ["":["R1"] -> ["R1"]
    response = _MALFORMED_ARRAY_OBJ_RE.sub('[', response)
    # Fix missing opening quote on keys:  ,attribute_errors"  ->  ,"attribute_errors"
    response = _MISSING_OPEN_QUOTE_RE.sub(r'\1"\2', response)
    # 1. Raw
    try:
        return _loads(response)
    except json.JSONDecodeError:
        pass
    # 2. Orphan keys (bare strings in object context, e.g. "pred_old")
    deorphaned = _fix_orphan_keys(response)
    try:
        return _loads(deorphaned)
    except json.JSONDecodeError:
        pass
    # 3. Invalid backslash escapes
    escaped = _INVALID_ESCAPE_RE.sub(r'\\\\', response)
    try:
        return _loads(escaped)
    except json.JSONDecodeError:
        pass
    # 4. Unescaped quotes (try on both raw and escape-fixed variants)
    for text in (response, escaped):
        try:
            return _fix_unescaped_quotes(text)
        except (json.JSONDecodeError, ValueError):
            pass
    ctx = f" for batch item {batch_idx}" if batch_idx is not None else ""
    raise ValueError(
        f"Failed to parse model response as JSON{ctx}\n"
        f"Response ({len(response)} chars):\n{response}"
    )

# ---------------------------------------------------------------------------
# vLLM model resolution
# ---------------------------------------------------------------------------

def resolve_model_for_vllm(model_name: str) -> str:
    """Resolve a HF model name to a local path suitable for vLLM.

    Works around a known issue where a stale
    ``model.safetensors.index.json`` on HuggingFace Hub references
    sharded weight files that don't exist alongside a consolidated
    ``model.safetensors``.  vLLM's weight loader trusts the index and
    ends up finding zero matching files.

    If the model is already a local path, it is returned unchanged.
    """
    if os.path.isdir(model_name):
        return model_name

    from huggingface_hub import snapshot_download
    snapshot_dir = snapshot_download(model_name)

    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    consolidated_path = os.path.join(snapshot_dir, "model.safetensors")
    if os.path.exists(index_path) and os.path.exists(consolidated_path):
        # Check if the index references files that don't actually exist
        with open(index_path) as f:
            index = json.load(f)
        referenced = set(index.get("weight_map", {}).values())
        missing = [
            fn for fn in referenced
            if not os.path.exists(os.path.join(snapshot_dir, fn))
        ]
        if missing:
            # Build a clean local copy without the stale index
            import tempfile
            clean_dir = os.path.join(
                tempfile.gettempdir(),
                f"vllm-{model_name.replace('/', '--')}",
            )
            os.makedirs(clean_dir, exist_ok=True)
            for entry in os.listdir(snapshot_dir):
                if entry == "model.safetensors.index.json":
                    continue
                dst = os.path.join(clean_dir, entry)
                if not os.path.exists(dst):
                    src = os.path.realpath(os.path.join(snapshot_dir, entry))
                    os.symlink(src, dst)
            return clean_dir

    return snapshot_dir