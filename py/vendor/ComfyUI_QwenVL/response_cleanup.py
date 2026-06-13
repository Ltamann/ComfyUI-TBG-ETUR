import re


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TRAILER_RE = re.compile(r".*</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_CAPTURE_RE = re.compile(r"<think\b[^>]*>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def strip_thinking(text, log_prefix=None):
    if text is None:
        return text
    cleaned = str(text).strip()
    if not cleaned:
        return cleaned

    thinking = []
    thinking.extend(part.strip() for part in _THINK_CAPTURE_RE.findall(cleaned) if part.strip())

    # Some Qwen thinking outputs lose the opening tag after special-token cleanup
    # but keep the closing tag; in that case the answer is after the final close.
    if re.search(r"</think\s*>", cleaned, flags=re.IGNORECASE):
        before_close = re.split(r"</think\s*>", cleaned, flags=re.IGNORECASE)[0].strip()
        before_close = re.sub(r"^<think\b[^>]*>", "", before_close, flags=re.IGNORECASE).strip()
        if before_close and before_close not in thinking:
            thinking.append(before_close)
        cleaned = _THINK_TRAILER_RE.sub("", cleaned).strip()

    cleaned = _THINK_BLOCK_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"^\s*(final answer|answer|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

    if thinking and log_prefix:
        print(f"{log_prefix} stripped thinking from response: {' '.join(thinking)}")
    return cleaned
