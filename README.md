# DSC thermal-pathway reconstruction benchmark

Reproducibility package for:

> Olga Lavinda. *Thermal pathway reconstruction exposes a reliability gap for AI-guided materials design.*

This repository supports a closed-book benchmark of source-qualified differential scanning calorimetry (DSC) event-map reconstruction. The primary cohort contains 30 molecular materials, with six polymers and one cocoa-butter case retained as separate descriptive strata.

The original API experiment recovered 299 of 534 molecular event identities. Terminal melting was recovered in 194 of 216 opportunities (89.8%), compared with 105 of 318 other thermal transitions and pathways (33.0%). A separate subscription-CLI extension evaluated two additional model/client configurations and a fixed paired protocol-context endpoint.

## Primary entry points

- `study/PROTOCOL.md`: original benchmark protocol.
- `study/ANALYSIS_PLAN.md`: prespecified analysis plan.
- `study/MODEL_PACKET.json`: blind prompt packet and output schema.
- `study/materials.json`: material definitions and cohort strata.
- `study/literature_reference.json`: source-qualified thermal-event reference.
- `data/original/`: scored original API outcomes and frozen result summaries.
- `data/extension/`: scored subscription-CLI extension outcomes and estimates.
- `derived/`: manuscript-level descriptive tables, verification record, and figure builders.
- `code/`: archived original and extension analysis implementations for method inspection.
- `PUBLIC_CLAIM_MAP.md`: claim-to-artifact map for the principal manuscript results.

## Reproduce the submitted descriptive results

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Then regenerate the four descriptive result tables, the verification record, and the three manuscript figures:

```bash
python3 derived/derive_submission_results.py
python3 derived/build_figures.py
```

The scripts resolve all inputs relative to this repository and require no private workspace paths. `derived/verification.json` records the source-file hashes and checks the reported numerators, denominators, paired contrasts, conditional direction coverage, and temperature summaries.

## Reproduction boundary

The repository supports scored-row-to-descriptive-result reproduction. It includes the structured event identities, states, directions, temperatures, assignments, and scoring labels required to audit the manuscript results. Raw provider responses, provider response identifiers, private reasoning traces, session metadata, and credentials are excluded.

The archived scripts under `code/` document the full scoring implementations but retain source-tree assumptions from the protected collection workspace. The portable public reproduction entry points are the two scripts under `derived/`.

The original API experiment and subscription-CLI extension remain separate in the data and analysis. Repeated generations are repeated opportunities within materials, not independent material samples. Event-family summaries are post hoc and descriptive.

## Manuscript and archival links

- ChemRxiv preprint: submitted; record link will be added when available.
- Dataset: [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.23004000).
- Analysis code and reproducibility documentation: this repository.

Repository: `https://github.com/LavindaLab/dsc-thermal-pathway-benchmark`

## Source and use information

Literature sources are identified by DOI and source locator in `study/literature_reference.json`. See `LICENSE` and `DATA_USE_NOTICE.md` for the boundary between author-generated code, structured benchmark records, model-derived fields, and third-party source material. File hashes for the release are recorded in `SHA256SUMS.txt`.
