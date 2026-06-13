"""
Step 8 — Observability layer.
Aggregates token usage, loop count, corpus references, confidence scores,
and latency across the full run into a structured report.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class IterationSummary:
    iteration: int
    word_count: int
    confidence: float
    corpus_queries: int
    corpus_ids_fetched: list[int]
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class ObservabilityReport:
    # Request metadata
    run_id: str
    timestamp: str
    alignment: str
    topic: str
    desired_length_words: int

    # Outcome
    success: bool
    blocked_reason: Optional[str]
    final_word_count: int
    final_confidence: float
    truncated: bool

    # Loop stats
    iterations_used: int
    max_iterations: int
    exit_reason: str          # "confidence_met" | "iterations_exhausted" | "blocked"

    # Token totals
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int

    # Corpus
    unique_corpus_ids: list[int]
    total_corpus_queries: int

    # Latency
    total_latency_ms: float
    avg_latency_per_iter_ms: float

    # Per-iteration breakdown
    iterations: list[IterationSummary] = field(default_factory=list)

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self) -> None:
        sep = "=" * 60
        print(sep)
        print(f"  OBSERVABILITY REPORT  |  run_id={self.run_id}")
        print(sep)
        print(f"  timestamp        : {self.timestamp}")
        print(f"  alignment        : {self.alignment}")
        print(f"  topic            : {self.topic}")
        print(f"  desired length   : {self.desired_length_words} words")
        print()
        print(f"  success          : {self.success}")
        if self.blocked_reason:
            print(f"  blocked reason   : {self.blocked_reason}")
        print(f"  final word count : {self.final_word_count}")
        print(f"  final confidence : {self.final_confidence:.3f}")
        print(f"  truncated        : {self.truncated}")
        print()
        print(f"  iterations used  : {self.iterations_used} / {self.max_iterations}")
        print(f"  exit reason      : {self.exit_reason}")
        print()
        print(f"  total tokens     : {self.total_tokens}  "
              f"({self.total_input_tokens} in / {self.total_output_tokens} out)")
        print(f"  corpus queries   : {self.total_corpus_queries}")
        print(f"  corpus IDs used  : {self.unique_corpus_ids}")
        print()
        print(f"  total latency    : {self.total_latency_ms:.0f} ms")
        print(f"  avg / iteration  : {self.avg_latency_per_iter_ms:.0f} ms")
        print()
        print("  per-iteration breakdown:")
        for it in self.iterations:
            print(
                f"    iter {it.iteration}: words={it.word_count:4d}  "
                f"conf={it.confidence:.2f}  "
                f"corpus_q={it.corpus_queries}  "
                f"tokens=({it.input_tokens}in/{it.output_tokens}out)  "
                f"latency={it.latency_ms:.0f}ms"
            )
        print(sep)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_report(
    *,
    run_id: str,
    alignment: str,
    topic: str,
    desired_length_words: int,
    success: bool,
    blocked_reason: Optional[str],
    final_word_count: int,
    final_confidence: float,
    truncated: bool,
    iterations_used: int,
    max_iterations: int,
    records: list,           # list[IterationRecord] from agent_loop
    corpus_speeches_used: list[int],
) -> ObservabilityReport:

    # Determine exit reason
    if not success:
        exit_reason = "blocked"
    elif iterations_used < max_iterations:
        exit_reason = "confidence_met"
    else:
        exit_reason = "iterations_exhausted"

    # Aggregate tokens and latency
    total_input = sum(r.token_usage["input"] for r in records)
    total_output = sum(r.token_usage["output"] for r in records)
    total_latency = sum(r.latency_ms for r in records)
    total_corpus_queries = sum(len(r.tool_calls) for r in records)

    # Per-iteration summaries
    iter_summaries = [
        IterationSummary(
            iteration=r.iteration,
            word_count=len(r.speech_segment.split()),
            confidence=r.confidence,
            corpus_queries=len(r.tool_calls),
            corpus_ids_fetched=[id_ for tc in r.tool_calls for id_ in tc.get("result_ids", [])],
            input_tokens=r.token_usage["input"],
            output_tokens=r.token_usage["output"],
            latency_ms=r.latency_ms,
        )
        for r in records
    ]

    return ObservabilityReport(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        alignment=alignment,
        topic=topic,
        desired_length_words=desired_length_words,
        success=success,
        blocked_reason=blocked_reason,
        final_word_count=final_word_count,
        final_confidence=final_confidence,
        truncated=truncated,
        iterations_used=iterations_used,
        max_iterations=max_iterations,
        exit_reason=exit_reason,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        unique_corpus_ids=corpus_speeches_used,
        total_corpus_queries=total_corpus_queries,
        total_latency_ms=total_latency,
        avg_latency_per_iter_ms=total_latency / max(iterations_used, 1),
        iterations=iter_summaries,
    )
