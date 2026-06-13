"""
Step 3 — Agent loop engine.
Drives iterative speech generation, checks confidence threshold,
and enforces the max iteration ceiling.

Step 2 (guardrails) gates every request before generation.
Step 4 (policy corpus) is exposed as an LLM tool — the model decides
when to query it and what arguments to pass.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import anthropic

from guardrails import GuardrailsRequest, run_guardrails
from observability import ObservabilityReport, build_report
from output import FinalOutput, finalize_output
from policy_corpus import init_db, query_speeches, seed_db
from telemetry import (
    blocked_counter,
    confidence_histogram,
    corpus_query_counter,
    iteration_histogram,
    latency_histogram,
    run_counter,
    token_counter,
    tracer,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.80
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Tool definition — exposed to the LLM
# ---------------------------------------------------------------------------

CORPUS_TOOL: anthropic.types.ToolParam = {
    "name": "search_policy_corpus",
    "description": (
        "Search a database of historical political speeches. "
        "Use this to find real examples of speeches that match a given topic, "
        "political alignment, or sentiment. The results give you stylistic reference "
        "material to improve the speech you are writing. "
        "Call this before drafting or whenever you need inspiration."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": (
                    "Topic keyword to search for, e.g. 'healthcare', 'climate', "
                    "'immigration', 'education'. Leave empty to search all topics."
                ),
            },
            "alignment": {
                "type": "string",
                "enum": ["left", "center-left", "center", "center-right", "right"],
                "description": "Filter by political alignment. Omit to return any alignment.",
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative", "inspirational", "cautionary"],
                "description": "Filter by emotional tone of the speech. Omit to return any sentiment.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of speeches to return (1–5). Defaults to 3.",
                "minimum": 1,
                "maximum": 5,
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

def _execute_corpus_tool(tool_input: dict) -> tuple[str, list[int]]:
    """
    Run the corpus query and return (formatted_result, list_of_ids).
    """
    speeches = query_speeches(
        topic=tool_input.get("topic"),
        alignment=tool_input.get("alignment"),
        sentiment=tool_input.get("sentiment"),
        limit=int(tool_input.get("limit", 3)),
    )

    if not speeches:
        return "No matching speeches found in the corpus.", []

    lines = []
    for sp in speeches:
        lines.append(
            f"ID {sp.id} | alignment={sp.alignment} | sentiment={sp.sentiment} | topic={sp.topic}\n"
            f"{sp.sample_text}"
        )
    return "\n\n---\n\n".join(lines), [sp.id for sp in speeches]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LoopRequest:
    alignment: str            # e.g. "left", "center-right"
    topic: str                # e.g. "healthcare policy reform"
    desired_length_words: int


@dataclass
class IterationRecord:
    iteration: int
    speech_segment: str
    confidence: float
    token_usage: dict
    latency_ms: float
    tool_calls: list[dict] = field(default_factory=list)   # corpus queries made this iter


@dataclass
class LoopResult:
    success: bool
    speech: str
    iterations_used: int
    confidence: float
    records: list[IterationRecord] = field(default_factory=list)
    blocked_reason: Optional[str] = None
    corpus_speeches_used: list[int] = field(default_factory=list)
    output: Optional[FinalOutput] = None          # step 7
    report: Optional[ObservabilityReport] = None  # step 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_system_prompt(alignment: str) -> str:
    return (
        f"You are an expert political speechwriter. "
        f"Your task is to write speeches that reflect a {alignment} political perspective. "
        f"Be factual, persuasive, and appropriate for a public audience. "
        f"Never produce hate speech, incitement, or content that violates ethical guidelines.\n\n"
        f"You have access to a policy corpus tool. Use it to search for historical speeches "
        f"on the topic before you start writing — they will help you match the right tone and style. "
        f"After gathering references, write the complete speech text and nothing else."
    )


def _build_user_prompt(
    topic: str,
    desired_length_words: int,
    previous_speech: Optional[str],
    iteration: int,
) -> str:
    if iteration == 1 or previous_speech is None:
        return (
            f"Write a political speech about: {topic}\n"
            f"Target length: approximately {desired_length_words} words.\n"
            f"Search the policy corpus first, then write only the speech text."
        )
    return (
        f"Revise and improve the following speech about '{topic}'. "
        f"Make it more compelling, coherent, and closer to {desired_length_words} words. "
        f"You may search the corpus again if helpful.\n\n"
        f"Previous draft:\n{previous_speech}\n\n"
        f"Return only the revised speech text."
    )


def _estimate_confidence(speech: str, desired_length_words: int) -> float:
    word_count = len(speech.split())
    length_score = 1.0 - min(abs(word_count - desired_length_words) / desired_length_words, 1.0)
    structure_keywords = ["today", "we must", "together", "in conclusion", "fellow", "our future"]
    structure_score = min(sum(kw in speech.lower() for kw in structure_keywords) / 3, 1.0)
    return round(0.6 * length_score + 0.4 * structure_score, 3)


# ---------------------------------------------------------------------------
# Single-iteration runner (handles tool-use turns internally)
# ---------------------------------------------------------------------------

def _run_one_iteration(
    client: anthropic.Anthropic,
    system_prompt: str,
    user_prompt: str,
    iteration: int,
) -> tuple[str, dict, float, list[dict], list[int]]:
    """
    Send a message to the LLM, handle any tool-use turns,
    then return (speech_text, token_usage, latency_ms, tool_calls, corpus_ids).
    """
    messages = [{"role": "user", "content": user_prompt}]
    total_tokens = {"input": 0, "output": 0}
    tool_calls_log: list[dict] = []
    all_corpus_ids: list[int] = []

    t0 = time.perf_counter()

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=[CORPUS_TOOL],
            messages=messages,
        )

        total_tokens["input"] += response.usage.input_tokens
        total_tokens["output"] += response.usage.output_tokens

        # ── Tool use: let the LLM query the corpus ──────────────────────────
        if response.stop_reason == "tool_use":
            # Append the assistant turn
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                tool_input = block.input
                print(
                    f"  [tool] search_policy_corpus("
                    + ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
                    + ")"
                )
                result_text, corpus_ids = _execute_corpus_tool(tool_input)
                all_corpus_ids.extend(corpus_ids)

                tool_calls_log.append({"tool": block.name, "input": tool_input, "result_ids": corpus_ids})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            # Feed results back and continue the loop
            messages.append({"role": "user", "content": tool_results})
            continue

        # ── Final text response ─────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t0) * 1000
        speech = next(
            (block.text.strip() for block in response.content if hasattr(block, "text")),
            "",
        )
        return speech, total_tokens, latency_ms, tool_calls_log, all_corpus_ids


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_agent_loop(request: LoopRequest) -> LoopResult:
    run_id = str(uuid.uuid4())[:8]
    labels = {"alignment": request.alignment}

    with tracer.start_as_current_span("harness.run") as root_span:
      root_span.set_attribute("run.id", run_id)
      root_span.set_attribute("run.alignment", request.alignment)
      root_span.set_attribute("run.topic", request.topic)
      root_span.set_attribute("run.desired_length_words", request.desired_length_words)
      return _run_agent_loop_inner(request, run_id, labels)


def _run_agent_loop_inner(request: LoopRequest, run_id: str, labels: dict) -> LoopResult:
    # ── Step 2: Guardrails ──────────────────────────────────────────────────
    outcome = run_guardrails(GuardrailsRequest(
        alignment=request.alignment,
        topic=request.topic,
        desired_length_words=request.desired_length_words,
    ))
    if not outcome.passed:
        run_counter.add(1, {**labels, "result": "blocked"})
        blocked_counter.add(1, {**labels, "rule": outcome.triggered_rule or "unknown"})
        report = build_report(
            run_id=run_id,
            alignment=request.alignment,
            topic=request.topic,
            desired_length_words=request.desired_length_words,
            success=False,
            blocked_reason=outcome.reason,
            final_word_count=0,
            final_confidence=0.0,
            truncated=False,
            iterations_used=0,
            max_iterations=MAX_ITERATIONS,
            records=[],
            corpus_speeches_used=[],
        )
        return LoopResult(
            success=False, speech="", iterations_used=0, confidence=0.0,
            blocked_reason=outcome.reason, report=report,
        )

    # ── Step 4: Ensure corpus DB is ready ───────────────────────────────────
    init_db()
    seed_db()

    # ── Step 3: Agent loop ──────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    system_prompt = _build_system_prompt(request.alignment)

    speech = ""
    confidence = 0.0
    records: list[IterationRecord] = []
    all_corpus_ids: list[int] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        user_prompt = _build_user_prompt(
            request.topic, request.desired_length_words, speech or None, iteration
        )

        with tracer.start_as_current_span(f"harness.iteration.{iteration}") as iter_span:
            iter_span.set_attribute("iteration.number", iteration)

            speech, token_usage, latency_ms, tool_calls, corpus_ids = _run_one_iteration(
                client, system_prompt, user_prompt, iteration
            )
            all_corpus_ids.extend(corpus_ids)
            confidence = _estimate_confidence(speech, request.desired_length_words)

            # trace attributes
            iter_span.set_attribute("iteration.word_count", len(speech.split()))
            iter_span.set_attribute("iteration.confidence", confidence)
            iter_span.set_attribute("iteration.input_tokens", token_usage["input"])
            iter_span.set_attribute("iteration.output_tokens", token_usage["output"])
            iter_span.set_attribute("iteration.latency_ms", latency_ms)
            iter_span.set_attribute("iteration.corpus_queries", len(tool_calls))

            # metrics
            token_counter.add(token_usage["input"] + token_usage["output"], labels)
            corpus_query_counter.add(len(tool_calls), labels)

            records.append(IterationRecord(
                iteration=iteration,
                speech_segment=speech,
                confidence=confidence,
                token_usage=token_usage,
                latency_ms=round(latency_ms, 1),
                tool_calls=tool_calls,
            ))

            print(
                f"[iter {iteration}/{MAX_ITERATIONS}] "
                f"words={len(speech.split())} "
                f"confidence={confidence:.2f} "
                f"corpus_queries={len(tool_calls)} "
                f"tokens=({token_usage['input']}in/{token_usage['output']}out) "
                f"latency={latency_ms:.0f}ms"
            )

            # ── Step 6: Exit condition ──────────────────────────────────────
            word_count = len(speech.split())
            length_met = abs(word_count - request.desired_length_words) / request.desired_length_words < 0.10
            if length_met and confidence >= CONFIDENCE_THRESHOLD:
                print(f"[loop] Exit: confidence {confidence:.2f} >= {CONFIDENCE_THRESHOLD} and length met.")
                break
    else:
        print(f"[loop] Exit: max iterations ({MAX_ITERATIONS}) reached.")

    deduped_corpus_ids = list(dict.fromkeys(all_corpus_ids))

    # ── End-of-run metrics ───────────────────────────────────────────────────
    total_latency = sum(r.latency_ms for r in records)
    run_counter.add(1, {**labels, "result": "success"})
    iteration_histogram.record(len(records), labels)
    latency_histogram.record(total_latency, labels)
    confidence_histogram.record(confidence, labels)

    # ── Step 7: Final output — validate, align, length-cap ──────────────────
    final = finalize_output(
        speech=speech,
        alignment=request.alignment,
        topic=request.topic,
        desired_length_words=request.desired_length_words,
    )

    if not final.validation_passed:
        print(f"[output] Post-generation block: {final.blocked_reason}")

    # ── Step 8: Observability report ─────────────────────────────────────────
    report = build_report(
        run_id=run_id,
        alignment=request.alignment,
        topic=request.topic,
        desired_length_words=request.desired_length_words,
        success=final.validation_passed,
        blocked_reason=final.blocked_reason,
        final_word_count=final.word_count,
        final_confidence=confidence,
        truncated=final.truncated,
        iterations_used=len(records),
        max_iterations=MAX_ITERATIONS,
        records=records,
        corpus_speeches_used=deduped_corpus_ids,
    )

    return LoopResult(
        success=final.validation_passed,
        speech=final.speech,
        iterations_used=len(records),
        confidence=confidence,
        records=records,
        corpus_speeches_used=deduped_corpus_ids,
        output=final,
        report=report,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_agent_loop(LoopRequest(
        alignment="center-left",
        topic="healthcare affordability",
        desired_length_words=300,
    ))

    if not result.success:
        print(f"\nBLOCKED: {result.blocked_reason}")
    else:
        print(f"\n{'='*60}")
        print(f"FINAL SPEECH  ({result.output.word_count} words)")
        print("=" * 60)
        print(result.speech)

    if result.report:
        print()
        result.report.print_summary()
