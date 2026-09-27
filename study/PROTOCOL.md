# Blinded ChemRxiv DSC benchmark protocol

## Scientific question

How accurately do the two frozen Luna API model arms recover the identities and
ordered pathways of source-qualified DSC events, and how does recovery vary
across prespecified event-complexity classes?

## Benchmark strata

- 30 unique pure small-molecule polymorphic compounds form the primary
  inference population.
- Six unique, source-specified polymer systems form a separate P1–P3
  complexity ladder.
- Ivory Coast cocoa butter is retained as one separate multicomponent case and
  is excluded from pure-compound pooled inference.

The authoritative cohort and gold files are `cohort/materials.json` and
`cohort/literature_reference.json`. The tracked root `materials.json` and the
external six-material pilot are historical design context; they are excluded
from request construction and benchmark inference.

## Frozen API design

- Responses API endpoint `POST /v1/responses`.
- Exact requested identifiers: `gpt-5.6-luna` and `gpt-6-luna`.
- Three fresh generations for each model–material cell: 37 × 2 × 3 = 222
  planned requests.
- Standard processing is forced with `service_tier="default"`; `store=false`,
  no tools or retrieval, medium reasoning effort, medium text verbosity, one
  strict JSON schema, and a 4,000-token output ceiling apply to every arm.
- Temperature is omitted from every request because parameter support may
  differ across reasoning models.
- Execution order is deterministic from seed 20260926 and balanced to within
  one request per model in every replicate-block quartile.
- One retry is allowed only for the frozen transient HTTP statuses or a
  transport failure; retries preserve the same scientific generation and all
  attempts are retained.

`blind/MODEL_PACKET.json` is the sole request-construction input.
`run_api_dsc.py` validates it offline by default. Collection mode requires
both the explicit `--execute` flag and a finite positive `--max-cost-usd`.
No credential is included in this bundle.

Cost accounting uses the standard short-context rates listed by the official
OpenAI model pages on 2026-09-26: $0.20/$1.20 per million input/output tokens
for `gpt-5.6-luna` and $0.10/$0.50 for `gpt-6-luna`. The runner treats every byte
of the serialized request as an input token and reserves the full 4,000-token
output ceiling for both allowed attempts. The resulting complete-run bound is
$1.7043; a $2 hard accounting ceiling covers that bound. Known usage replaces
the reservation after each terminal record, while attempts lacking usage remain
charged at their full bound.

Before each request, the runner writes an in-flight receipt. A received response
is durably staged before it enters the append-only terminal ledger. On restart,
a staged response is recovered without another API call; an ambiguous in-flight
receipt stops execution for reconciliation rather than duplicating a scientific
generation.

Pricing sources: [GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
and [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

## Blinding and preservation

Prompts include the material identity, neutral specimen description, target
starting forms or states, reconstruction instructions, and the shared output
schema. They contain no citation, locator, reference temperature, gold event
assignment, expected event count, or material-specific pathway hint. The gold
file is never read by the runner.

Every terminal ledger record retains the exact prompt and hash, requested and
returned model identifiers, response ID, UTC timestamp, settings, usage,
completion state, incomplete reason, error, attempt number, raw output array,
and exact raw response JSON. Failed and retried attempts remain append-only.

## Outcomes and inference

The primary estimand is correct event-identity recovery across M1–M3. The
material is the independent cluster; forms, events, models, and generations are
nested. Temperature error is evaluated only after correct identity and a
compatible temperature field. Common terminal melts reached by multiple
precursors share one scoring unit and are not multiplied.

The complete matching, inference, fallback, multiplicity, missing-output, and
sensitivity rules are frozen in `ANALYSIS_PLAN.md`.

## Execution boundary

This freeze authorizes validation only. It does not authorize benchmark API
calls, publication, upload, repository push, submission, purchase, credential
change, or external message.
