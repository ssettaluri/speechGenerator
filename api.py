"""
REST API — wraps run_agent_loop so any HTTP client can generate speeches.

Endpoints:
  POST /generate         — run the full harness pipeline
  GET  /health           — liveness check
  GET  /alignments       — list valid alignment values
  GET  /runs/{run_id}    — retrieve a past run from the in-memory store
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent_loop import LoopRequest, run_agent_loop

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Speech Generator API",
    description="Generates politically-aligned speeches using the harness pipeline (guardrails → agent loop → policy corpus → final output).",
    version="1.0.0",
)

# In-memory run store  {run_id: report_dict}
_run_store: dict[str, dict] = {}

VALID_ALIGNMENTS = ["left", "center-left", "center", "center-right", "right"]

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    alignment: str = Field(
        ...,
        description="Political alignment of the speech.",
        examples=["center-left"],
    )
    topic: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Topic the speech should address.",
        examples=["healthcare affordability"],
    )
    desired_length_words: int = Field(
        ...,
        ge=50,
        le=5000,
        description="Target word count for the final speech.",
        examples=[300],
    )

    @field_validator("alignment")
    @classmethod
    def validate_alignment(cls, v: str) -> str:
        if v not in VALID_ALIGNMENTS:
            raise ValueError(f"alignment must be one of: {', '.join(VALID_ALIGNMENTS)}")
        return v


class IterationOut(BaseModel):
    iteration: int
    word_count: int
    confidence: float
    corpus_queries: int
    corpus_ids_fetched: list[int]
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ObservabilityOut(BaseModel):
    run_id: str
    timestamp: str
    exit_reason: str
    iterations_used: int
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_corpus_queries: int
    unique_corpus_ids: list[int]
    total_latency_ms: float
    avg_latency_per_iter_ms: float
    iterations: list[IterationOut]


class GenerateResponse(BaseModel):
    success: bool
    run_id: str
    speech: Optional[str]
    word_count: int
    confidence: float
    truncated: bool
    blocked_reason: Optional[str]
    observability: ObservabilityOut


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.get("/alignments", tags=["Meta"])
def alignments():
    """Return all valid alignment values."""
    return {"alignments": VALID_ALIGNMENTS}


@app.post("/generate", response_model=GenerateResponse, tags=["Speech"])
def generate(body: GenerateRequest):
    """
    Run the full harness pipeline and return the generated speech plus
    an observability report for the run.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set on the server.")

    result = run_agent_loop(LoopRequest(
        alignment=body.alignment,
        topic=body.topic,
        desired_length_words=body.desired_length_words,
    ))

    report = result.report
    obs = ObservabilityOut(
        run_id=report.run_id,
        timestamp=report.timestamp,
        exit_reason=report.exit_reason,
        iterations_used=report.iterations_used,
        total_tokens=report.total_tokens,
        total_input_tokens=report.total_input_tokens,
        total_output_tokens=report.total_output_tokens,
        total_corpus_queries=report.total_corpus_queries,
        unique_corpus_ids=report.unique_corpus_ids,
        total_latency_ms=report.total_latency_ms,
        avg_latency_per_iter_ms=report.avg_latency_per_iter_ms,
        iterations=[IterationOut(**vars(it)) for it in report.iterations],
    )

    response = GenerateResponse(
        success=result.success,
        run_id=report.run_id,
        speech=result.speech or None,
        word_count=result.output.word_count if result.output else 0,
        confidence=result.confidence,
        truncated=result.output.truncated if result.output else False,
        blocked_reason=result.blocked_reason,
        observability=obs,
    )

    # Cache the run
    _run_store[report.run_id] = response.model_dump()

    return response


@app.get("/runs/{run_id}", response_model=GenerateResponse, tags=["Speech"])
def get_run(run_id: str):
    """Retrieve a previously generated speech by run ID."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return _run_store[run_id]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
