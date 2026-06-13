
1. Human input — the user specifies political alignment, topic, and desired length.
2. Guardrails — every request passes through the filter layer first (hate speech detection, banned topics, system prompt rules) before anything is generated.
3. Agent loop engine — the core orchestrator that drives iterative generation, checks the confidence threshold, and enforces the max iteration ceiling.
4. Policy corpus tool — the loop calls out to retrieve historical speeches by topic/sentiment; results feed back in as context.
5. LLM — generates each speech segment; output returns to the loop for evaluation.
6. Exit condition — the loop terminates when length is met, confidence clears the threshold, or iterations are exhausted.
7. Final output — the validated, aligned, length-capped speech.
8. Observability layer — captures token usage, loop count, source speeches referenced, confidence scores, and latency across the full run.