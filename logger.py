"""
Structured JSON logger for the speech-generator harness.
Every event written to console is also appended as a JSON line to speech_generator.log.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.environ.get("LOG_PATH", Path(__file__).parent / "speech_generator.log"))

# ---------------------------------------------------------------------------
# Standard Python logger (writes plain text to stderr + JSON to file)
# ---------------------------------------------------------------------------

_file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(message)s"))  # raw JSON lines

_stderr_handler = logging.StreamHandler()
_stderr_handler.setLevel(logging.WARNING)  # only warnings+ to stderr (rich handles the rest)

logging.basicConfig(handlers=[_file_handler, _stderr_handler], level=logging.DEBUG)
_log = logging.getLogger("speech_generator")


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _write(event: str, level: str = "INFO", **fields):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    _log.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Public log functions — mirror every console.py call
# ---------------------------------------------------------------------------

def log_run_start(run_id: str, alignment: str, topic: str, desired_length_words: int):
    _write("run_start", run_id=run_id, alignment=alignment,
           topic=topic, desired_length_words=desired_length_words)


def log_guardrails_pass(run_id: str = ""):
    _write("guardrails_pass", run_id=run_id)


def log_guardrails_block(run_id: str, rule: str, reason: str):
    _write("guardrails_block", level="WARNING", run_id=run_id, rule=rule, reason=reason)


def log_corpus_query(run_id: str, iteration: int, tool_input: dict, result_count: int):
    _write("corpus_query", run_id=run_id, iteration=iteration,
           query=tool_input, result_count=result_count)


def log_corpus_cap(run_id: str, iteration: int, limit: int):
    _write("corpus_cap_reached", level="WARNING", run_id=run_id,
           iteration=iteration, limit=limit)


def log_iteration(run_id: str, iteration: int, max_iter: int, word_count: int,
                  desired: int, confidence: float, corpus_queries: int,
                  input_tokens: int, output_tokens: int, latency_ms: float):
    _write(
        "iteration_complete",
        run_id=run_id,
        iteration=iteration,
        max_iterations=max_iter,
        word_count=word_count,
        desired_length_words=desired,
        confidence=round(confidence, 4),
        corpus_queries=corpus_queries,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=round(latency_ms, 1),
    )


def log_exit(run_id: str, reason: str):
    level = "INFO" if reason == "confidence_met" else "WARNING"
    _write("loop_exit", level=level, run_id=run_id, reason=reason)


def log_output_block(run_id: str, reason: str):
    _write("output_blocked", level="WARNING", run_id=run_id, reason=reason)


def log_run_complete(run_id: str, success: bool, word_count: int,
                     confidence: float, truncated: bool,
                     iterations_used: int, total_tokens: int,
                     total_latency_ms: float, corpus_ids: list[int],
                     speech_preview: str):
    _write(
        "run_complete",
        level="INFO" if success else "WARNING",
        run_id=run_id,
        success=success,
        word_count=word_count,
        confidence=round(confidence, 4),
        truncated=truncated,
        iterations_used=iterations_used,
        total_tokens=total_tokens,
        total_latency_ms=round(total_latency_ms, 1),
        corpus_ids_used=corpus_ids,
        speech_preview=speech_preview[:120] + "..." if len(speech_preview) > 120 else speech_preview,
    )
