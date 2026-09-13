# Strict manual scoring rubric

- RootCauseCorrectness: 1 only when the predicted cause clearly matches the issue asked in the query logically and semantically; otherwise 0.
- EvidenceGrounding: 1 only when the explanation is supported by the retrieved logs; otherwise 0.
- Hallucination: 1 when the answer introduces an unsupported causal claim; otherwise 0.

Use only binary values 0 or 1. `Hallucination = 1` means the answer hallucinated.
