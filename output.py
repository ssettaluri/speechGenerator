"""
Step 7 — Final output.
Validates, aligns, and hard-caps the generated speech before it leaves the system.

Three guarantees:
  1. Length cap   — truncates to desired_length_words at a sentence boundary.
  2. Alignment    — runs a post-generation guardrails pass on the *output* text
                    (catches anything the LLM may have slipped in).
  3. Sanitisation — strips any stray system/tool artefacts from the text.
"""

import re
from dataclasses import dataclass
from typing import Optional

from guardrails import GuardrailsRequest, run_guardrails


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FinalOutput:
    speech: str
    word_count: int
    truncated: bool          # True if the text was cut to meet the length cap
    validation_passed: bool
    blocked_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ARTEFACT_PATTERNS = [
    r"\[tool[^\]]*\]",           # [tool_use] / [tool_result] tags
    r"<\|.*?\|>",                # <|special tokens|>
    r"ID \d+ \| alignment=\S+",  # raw corpus dump lines that leaked through
]


def _strip_artefacts(text: str) -> str:
    for pat in _ARTEFACT_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _truncate_to_words(text: str, max_words: int) -> tuple[str, bool]:
    """
    Truncate text to at most max_words words, breaking at the nearest
    sentence boundary ('. ', '! ', '? ') to avoid cutting mid-sentence.
    Returns (truncated_text, was_truncated).
    """
    words = text.split()
    if len(words) <= max_words:
        return text, False

    # Take a candidate window slightly over the limit and walk back to a sentence end
    candidate = " ".join(words[:max_words])
    for sep in (". ", "! ", "? "):
        idx = candidate.rfind(sep)
        if idx != -1:
            return candidate[: idx + 1].strip(), True

    # No sentence boundary found — hard cut
    return candidate.strip(), True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def finalize_output(
    speech: str,
    alignment: str,
    topic: str,
    desired_length_words: int,
) -> FinalOutput:
    """
    Step 7 pipeline:
      strip artefacts → hard length cap → post-generation guardrails check.
    """
    from telemetry import tracer
    with tracer.start_as_current_span("output.finalize") as span:
        span.set_attribute("output.input_word_count", len(speech.split()))
        span.set_attribute("output.desired_length_words", desired_length_words)

        # 1. Strip LLM/tool artefacts
        clean = _strip_artefacts(speech)

        # 2. Hard length cap at sentence boundary
        capped, truncated = _truncate_to_words(clean, desired_length_words)
        span.set_attribute("output.truncated", truncated)
        span.set_attribute("output.final_word_count", len(capped.split()))

        # 3. Post-generation guardrails on the *output* text
        #    Reuse the same filter layer with the output text as the topic field
        #    so hate-speech / banned-topic checks run over what was actually generated.
        outcome = run_guardrails(GuardrailsRequest(
            alignment=alignment,
            topic=capped,               # check the generated content, not just the input topic
            desired_length_words=desired_length_words,
        ))

        if not outcome.passed:
            span.set_attribute("output.validation_passed", False)
            span.set_attribute("output.blocked_reason", outcome.reason or "")
            return FinalOutput(
                speech="",
                word_count=0,
                truncated=truncated,
                validation_passed=False,
                blocked_reason=f"Post-generation filter: {outcome.reason}",
            )

        span.set_attribute("output.validation_passed", True)
        return FinalOutput(
            speech=capped,
            word_count=len(capped.split()),
            truncated=truncated,
            validation_passed=True,
        )


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = (
        "Fellow citizens, today we stand at a crossroads. "
        "We must choose between a future of hope and one of fear. "
        "Together, our future is bright. In conclusion, let us march forward. "
        "Extra word " * 50  # simulate over-length output
    )

    out = finalize_output(
        speech=sample,
        alignment="center-left",
        topic="healthcare",
        desired_length_words=30,
    )

    print(f"validation_passed : {out.validation_passed}")
    print(f"truncated         : {out.truncated}")
    print(f"word_count        : {out.word_count}")
    print(f"speech            : {out.speech}")
