# Public claim-to-artifact map

| Manuscript result | Evidence record | Reproduction or verification code |
|---|---|---|
| Original molecular identity recall: 299/534 overall, 194/216 terminal melting, 105/318 other transitions/pathways | `data/original/event_matches.csv`; `data/original/gold_outcomes.csv`; `data/original/primary_results.json` | `derived/derive_submission_results.py`; `derived/verification.json` |
| Broad-prompt event-family recall across four configurations | `data/original/event_matches.csv`; `data/extension/event_matches.csv`; `derived/event_family_recall.csv` | `derived/derive_submission_results.py`; `derived/build_figures.py` |
| Fixed paired protocol endpoint: Sol 53/123 to 45/123; Opus 79/123 to 73/123 | `data/extension/paired_gold_outcomes.csv`; `derived/paired_protocol_recall.csv` | `derived/derive_submission_results.py`; `code/analyze_extension.py` |
| Conditional direction coverage and temperature summaries | `data/extension/event_matches.csv`; `derived/conditional_direction_temperature.csv` | `derived/derive_submission_results.py`; `derived/build_figures.py` |
| Acetaminophen, carbamazepine, and flufenamic-acid pathway examples | `study/literature_reference.json`; `derived/illustrative_pathways.csv` | `derived/derive_submission_results.py` |
| Cohort composition and separation of molecular, polymer, and cocoa-butter strata | `study/materials.json`; `study/literature_reference.json`; `data/original/stratum_results.csv` | `code/analyze_dsc_benchmark.py` |

The manuscript-level event-family and conditional-measurement summaries are post hoc descriptive analyses. The original and extension collections retain their separate model/client and study-design boundaries.
