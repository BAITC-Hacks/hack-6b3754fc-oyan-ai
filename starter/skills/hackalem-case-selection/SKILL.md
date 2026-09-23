---
name: hackalem-case-selection
description: Select one newly published HackAlem case within 15 minutes using its exact requirements, rubric, feasibility, and demo risk. Use when official cases arrive; do not use to implement the chosen case.
---

# HackAlem case selection

Produce a defensible case decision quickly without writing product code.

## Inputs

Require the exact case text, mandatory outputs, scoring rubric, provided data/assets, allowed services, and submission checks. Mark missing information as unknown; never fill gaps from assumption.

## Workflow

1. Put all candidate cases into `docs/CASE_SELECTION.md` with their exact titles and links.
2. Apply these hard gates before scoring:
   - mandatory inputs and services are accessible now;
   - the main scenario can be built and verified in five hours;
   - experts can run it without a participant's personal account;
   - there is a deterministic demo or fallback;
   - licenses, data rights, and disclosure requirements can be satisfied.
3. If multi-agent tools are available, use no more than three read-only workstreams:
   - extract exact requirements and rubric coverage;
   - test data/API/SDK feasibility from primary sources;
   - act as a skeptical judge and identify demo and differentiation risks.
4. Score every surviving case from 0-5 and apply the weights in `docs/CASE_SELECTION.md`. Cite evidence for non-obvious scores; label estimates.
5. Prefer the case with the strongest executable evidence, not the most exciting concept. Break close ties by lower integration risk and clearer judge-visible result.
6. Record one selected case, the decisive reason, rejected alternatives, confidence, and the first risk-reduction test.
7. Copy the selected case verbatim into `docs/CHALLENGE_BRIEF.md`. Stop before implementation until the hard requirements, acceptance checks, and first vertical slice are explicit.

Do not browse for solution ideas until requirements have been extracted. Do not let speculative market size compensate for missing data, credentials, or a weak demo path.
