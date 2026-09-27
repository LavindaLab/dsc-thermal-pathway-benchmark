# Prespecified analysis plan

**Study ID:** DSC-CHEMRXIV-COHORT-2026-09-26  
**Frozen:** 2026-09-26, before benchmark collection

## Analysis populations

The molecular-polymorph cohort is the primary analysis population. Polymer systems and cocoa butter are separate prospective controls/case studies. Every requested generation belongs to the intention-to-benchmark set, including failures, refusals, and incomplete responses. The parseable set is secondary and cannot replace the intention-to-benchmark denominator.

The material is the independent cluster. Gold events, predicted events, starting forms, and the three generations are nested observations.

## Canonical event representation

Gold and predicted events are represented by:

`material → starting form/state → event family → transformation or endpoint identity → direction → temperature field`

Event families are `glass_transition`, `crystallization_on_cooling`, `cold_crystallization`, `solid_solid_transition`, `melt_mediated_conversion`, `melt_sublimation_recrystallization`, `melting`, and `recrystallization`. These names match the frozen gold file. No desolvation event is present in the eligible pure-compound cohort.

The response schema exposes every family above and also permits `other`. An `other` event receives no automatic family match and is sent to blinded adjudication; the parser may not infer a more specific family from temperature or the gold file.

The parser may normalize spelling, Greek-letter names, Roman numerals, Celsius/Kelvin conversion, and standard synonyms. It may not infer an event absent from the raw output or replace a model assignment with a gold assignment. Raw output remains immutable.

## One-to-one matching

Temperature is not used to decide event identity. A predicted event is eligible to match a gold event when material, starting form/state (or explicitly stated uncertainty set), event family, and transformation/endpoint identity are compatible. A one-to-one maximum-cardinality match is then chosen; ties minimize semantic disagreement before temperature is examined.

Assignments use three labels:

- **correct identity:** family and transformation/endpoint agree;
- **partial/ambiguous identity:** family agrees but the named starting or product form is absent, unresolved, or incompatible with source-level ambiguity;
- **incorrect identity:** family or transformation/endpoint conflicts.

Only correct identity is a true positive in the primary analysis. Partial matches are false negatives plus false positives in the primary analysis and receive credit only in a prespecified sensitivity analysis.

Two adjudicators, blinded to model arm and temperature error, independently review parser-flagged ambiguities. They see the raw statement, canonical candidate labels, and the source-qualified gold definition. Consensus resolves disagreements; unresolved cases keep the conservative primary classification. Automatic matches and all adjudication changes are logged.

## Repeated melts and pathway units

When several starting forms convert to the same terminal form under one protocol, the final melt is one material–protocol scoring unit. The same terminal melt is not multiplied by the number of precursor forms. Separate melt anchors are retained only for different terminal forms, different qualified protocols that the prompt distinguishes, or genuinely separate form-specific paths.

This rule applies before counts are frozen and to both gold and predicted event sets.

## Metrics

For each material–model–generation:

- recall = matched correct gold identities / qualified gold identities;
- precision = matched correct predicted identities / predicted identities;
- F1 = harmonic mean of recall and precision, defined as zero when predictions are present but none are correct;
- direction accuracy = correct endothermic/exothermic/baseline-shift direction / identity-correct matched events with a scorable direction.

An empty response has recall 0. Precision is undefined when no events are emitted and is summarized with an explicit empty-response indicator; for macro-F1 it contributes 0 when at least one gold event exists.

## Temperature accuracy

Temperature error is evaluated only after event identity is correct. A prediction must state or unambiguously imply the same field as the gold anchor: onset, midpoint, peak, interval, or one-sided bound.

- point versus point: absolute Celsius difference;
- point versus gold interval: zero inside the interval, otherwise distance to the nearest bound;
- predicted interval versus gold point: distance from the point to the interval, with zero inside;
- interval versus interval: zero if intervals overlap, otherwise nearest-bound distance;
- one-sided bound: distance beyond the bound, zero on the allowed side.

Kelvin is converted to Celsius before scoring. A peak is not silently compared with an onset or midpoint in the primary analysis. A sensitivity analysis allows field-agnostic matching and labels the resulting error separately.

Summaries report median absolute error, interquartile range, material-level mean absolute error, and 95% cluster-bootstrap intervals. Temperature performance never rescues a wrong identity.

## Confidence calibration

When the model supplies confidence, emitted events receive a binary correctness outcome after one-to-one matching. Calibration is described with reliability curves, Brier score, calibration intercept/slope where estimable, and expected calibration error using fixed bins. Missing events have no emitted confidence and are represented through recall, not imputed. Calibration analyses therefore condition on an event being emitted and are secondary.

## Primary model and clustering

The primary dataset has one row per unique molecular gold scoring unit × model × generation. The outcome is correct identity recovery. The model includes fixed effects for complexity class, model arm, generation, terminal-melt indicator, and model × complexity, with a random intercept for material. The primary complexity test is a joint Wald or likelihood-ratio test for M1–M3.

If the mixed model is singular or fails to converge under the frozen optimizer settings, the sole fallback is a logistic generalized estimating equation with material clusters and exchangeable working correlation. The fallback is reported as such; no additional model search is permitted.

Uncertainty for marginal metrics and contrasts uses 10,000 stratified cluster-bootstrap resamples. Materials are sampled with replacement within the molecular complexity classes; all nested events, models, and generations move together. Percentile intervals are primary, with BCa intervals as a sensitivity analysis if stable.

## Secondary comparisons and multiplicity

The joint complexity test is the sole primary significance test. Within the primary model, pairwise complexity contrasts and model contrasts use Holm adjustment within their respective families. The model × complexity interaction is secondary. Polymer-tier, cocoa-butter, confidence, temperature-field, and individual-material analyses are descriptive or secondary and do not support a global model-ranking claim.

Effect estimates and confidence intervals lead reporting. Exact p-values accompany prespecified tests; arbitrary significance labels do not determine scientific interpretation.

## Terminal melting and non-melting pathways

Every gold scoring unit is marked `terminal_melting` or `non_melting_pathway`. Recall, precision/F1 decomposition, and direction accuracy are reported for both. A secondary model includes event class and its interaction with complexity. Shared post-conversion melts remain deduplicated under the pathway rule above.

## Run-to-run stability

For the three generations within each model–material cell:

- pairwise Jaccard similarity and pairwise F1 of canonical event-identity sets;
- unanimity rate for each gold identity;
- within-cell standard deviation and range of temperature predictions for identities present in at least two runs; and
- parse/failure-status concordance.

These are summarized first at material level, then across material clusters.

## Missing, failed, and incomplete outputs

Authentication or transport failures, API refusals, incomplete responses, empty output, and invalid JSON are distinct statuses. A deterministic parser repair may recover syntactically malformed JSON without changing words or numbers; both raw and repaired representations are retained. A response that cannot be parsed after the frozen repair is scored as recovering no events in the intention-to-benchmark analysis.

Only frozen transient errors receive a retry. A retry replaces no raw record and does not create an extra scientific generation. Missing outputs are not multiply imputed. A parseable-output sensitivity analysis is labeled as selection-prone.

## Model-identifier integrity

Every response retains both the requested and returned model identifiers. A returned identifier inconsistent with the requested Luna generation pauses that arm for reconciliation. No output is silently relabeled or pooled across identifiers.

## Sensitivity analyses

Prespecified sensitivities are:

1. partial/ambiguous identity receives half credit;
2. field-agnostic temperature error after correct identity;
3. exclusion of gold anchors whose protocol has any `not_reported` field;
4. exclusion of events represented only as source intervals or bounds;
5. equal weighting of materials instead of gold events;
6. leave-one-material-out influence analysis;
7. BCa rather than percentile cluster-bootstrap intervals;
8. GEE fallback comparison when the mixed model converges;
9. exclusion of any response with an unresolved requested/returned identifier mismatch; and
10. parseable outputs only, explicitly separated from the primary intention-to-benchmark result.

No sensitivity analysis may redefine complexity after viewing model outputs.

## Quality controls

Before analysis, code must verify unique IDs, prompt hashes, three planned generations per model–material cell, exact model names, immutable raw outputs, and agreement between reported and machine-derived counts. Analysis code reads the gold reference and response ledger separately and joins them only after collection is complete. Analysts adjudicating identities remain blind to model arm until the adjudication table is frozen.
