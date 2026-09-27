#!/usr/bin/env python3
"""Analyze the frozen subscription-CLI extension with the original matcher."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_dsc_benchmark import (
    attach_conditional_scores,
    create_response_metrics,
    make_gold_outcomes,
    make_gold_units,
    parse_and_match,
    quantile,
    write_csv,
)


EXT = ROOT / "extension"
COLLECTION = EXT / "collection"
RESULTS = EXT / "results"
INTERNAL = EXT / "internal"
SEED = 20260927
BOOTSTRAPS = 10_000
MODELS = ("gpt-5.6-sol", "claude-opus-5-5")


class ExtensionAnalysisError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_schema_validate(value: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "top_level_not_object"
    properties = schema["properties"]
    if set(value) != set(schema["required"]) or set(value) - set(properties):
        return False, "top_level_fields_invalid"
    if not isinstance(value.get("material_id"), str) or not value["material_id"].strip():
        return False, "material_id_invalid"
    if value.get("schematic_reconstruction") is not True:
        return False, "schematic_flag_invalid"
    if not isinstance(value.get("protocol_assumptions"), list) or not all(isinstance(x, str) for x in value["protocol_assumptions"]):
        return False, "protocol_assumptions_invalid"
    if not isinstance(value.get("events"), list):
        return False, "events_invalid"
    event_schema = properties["events"]["items"]
    for index, event in enumerate(value["events"], 1):
        if not isinstance(event, dict) or set(event) != set(event_schema["required"]):
            return False, f"event_{index}_fields_invalid"
        if event["event_type"] not in event_schema["properties"]["event_type"]["enum"]:
            return False, f"event_{index}_type_invalid"
        if event["direction"] not in event_schema["properties"]["direction"]["enum"]:
            return False, f"event_{index}_direction_invalid"
        temperature = event.get("temperature")
        temperature_schema = event_schema["properties"]["temperature"]
        if not isinstance(temperature, dict) or set(temperature) != set(temperature_schema["required"]):
            return False, f"event_{index}_temperature_fields_invalid"
        if temperature["kind"] not in temperature_schema["properties"]["kind"]["enum"]:
            return False, f"event_{index}_temperature_kind_invalid"
        confidence = event.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            return False, f"event_{index}_confidence_invalid"
    return True, "valid"


def load_and_validate_collection() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    packet = load_json(EXT / "prompt_packet.json")
    ready = load_json(EXT / "READY_FOR_PREFLIGHT.json")
    approval = load_json(EXT / "ROOT_PRECOLLECTION_APPROVAL.json")
    manifest = load_json(COLLECTION / "collection_manifest.json")
    planned = load_json(COLLECTION / "planned_tasks.json")
    outcomes = load_jsonl(COLLECTION / "outcomes.jsonl")
    complete = load_json(EXT / "COLLECTION_COMPLETE.json")
    defects: list[str] = []

    if manifest["input_packet_sha256"] != sha256(EXT / "prompt_packet.json"):
        defects.append("collector packet hash differs from frozen prompt packet")
    if manifest["planned_tasks_sha256"] != sha256(COLLECTION / "planned_tasks.json"):
        defects.append("planned-task hash mismatch")
    if complete["outcomes_sha256"] != sha256(COLLECTION / "outcomes.jsonl"):
        defects.append("outcomes hash mismatch")
    for name, expected in approval["sha256"].items():
        if sha256(EXT / name) != expected:
            defects.append(f"approved preflight hash changed: {name}")
    for path, expected in ready["protected_original_evidence_sha256"].items():
        if sha256(ROOT / path) != expected:
            defects.append(f"protected evidence changed: {path}")

    planned_keys = [row["key"] for row in planned]
    outcome_keys = [row["key"] for row in outcomes]
    if len(planned) != 330 or len(set(planned_keys)) != 330:
        defects.append("planned task count/uniqueness is not 330")
    if set(outcome_keys) != set(planned_keys):
        defects.append("outcome keys differ from planned tasks")
    if len(outcomes) != 330 or len(set(outcome_keys)) != 330:
        defects.append("outcome count/uniqueness is not 330")
    if Counter(row["status"] for row in outcomes) != {"completed": 330}:
        defects.append("collection is not 330 completed outcomes")
    if complete != {
        "planned": 330,
        "attempted": 330,
        "completed": 330,
        "failed": 0,
        "finished": complete.get("finished"),
        "outcomes_sha256": complete.get("outcomes_sha256"),
    }:
        defects.append("completion receipt accounting is inconsistent")

    schema_failures: list[str] = []
    ledger: list[dict[str, Any]] = []
    outcomes_by_key = {row["key"]: row for row in outcomes}
    for task in planned:
        outcome = outcomes_by_key[task["key"]]
        if any(task[field] != outcome[field] for field in ("key", "material_id", "condition", "generation")):
            defects.append(f"task/outcome identity mismatch: {task['key']}")
            continue
        if task["model"] != outcome["model"]:
            defects.append(f"task/outcome model mismatch: {task['key']}")
        raw_path = ROOT / outcome["raw_path"]
        final_path = ROOT / outcome["final_path"]
        if not raw_path.exists() or not final_path.exists():
            defects.append(f"missing raw/final path: {task['key']}")
            continue
        if sha256(raw_path) != outcome["raw_sha256"]:
            defects.append(f"raw hash mismatch: {task['key']}")
        value = load_json(final_path)
        valid, status = exact_schema_validate(value, packet["schema"])
        if not valid:
            schema_failures.append(f"{task['key']}:{status}")
        returned_models = outcome.get("returned_models") or []
        if outcome["model"] == "claude-opus-5-5":
            if returned_models != ["claude-opus-5-5"]:
                defects.append(f"Opus returned identifier mismatch: {task['key']}")
            returned_model: str | None = "claude-opus-5-5"
        else:
            if returned_models:
                defects.append(f"Sol returned identifier unexpectedly exposed: {task['key']}")
            returned_model = None
        usage = dict(outcome.get("usage") or {})
        if "reasoning_output_tokens" in usage:
            usage["output_tokens_details"] = {"reasoning_tokens": int(usage["reasoning_output_tokens"] or 0)}
        usage.setdefault("total_tokens", int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0))
        output_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        ledger.append({
            "key": task["key"],
            "response_id": outcome["blind_id"],
            "material_id": task["material_id"],
            "requested_model": task["model"],
            "returned_model": returned_model,
            "generation": task["generation"],
            "condition": task["condition"],
            "completion_state": "completed",
            "incomplete_reason": None,
            "usage": usage,
            "raw_output": [{"type": "message", "content": [{"type": "output_text", "text": output_text}]}],
        })
    if schema_failures:
        defects.extend(schema_failures)
    if defects:
        raise ExtensionAnalysisError("Collection validation failed:\n- " + "\n- ".join(defects[:80]))
    return ledger, planned, packet


def eligibility_rows() -> list[dict[str, str]]:
    with (EXT / "eligibility.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def selected_units(rows: Sequence[dict[str, str]]) -> tuple[set[str], set[str], set[str]]:
    all_units: set[str] = set()
    nonmelting: set[str] = set()
    endpoint_materials: set[str] = set()
    for row in rows:
        if row["protocol_eligible"] != "yes":
            continue
        all_units.update(filter(None, row["fixed_reference_unit_ids_both_arms"].split("|")))
        local_nm = set(filter(None, row["fixed_non_melting_unit_ids_both_arms"].split("|")))
        nonmelting.update(local_nm)
        if local_nm:
            endpoint_materials.add(row["material_id"])
    return all_units, nonmelting, endpoint_materials


def add_condition(rows: Iterable[dict[str, Any]], condition_by_key: dict[str, str]) -> None:
    for row in rows:
        if "key" in row:
            row["condition"] = condition_by_key[row["key"]]


def aggregate_ratio_ci(
    rows: Sequence[dict[str, Any]],
    numerator: str,
    denominator: str,
    cluster: str = "material_id",
    cluster_universe: set[str] | None = None,
) -> dict[str, Any]:
    by_cluster: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        by_cluster[str(row[cluster])][0] += int(row[numerator])
        by_cluster[str(row[cluster])][1] += int(row[denominator])
    if cluster_universe is not None:
        for item in cluster_universe:
            by_cluster.setdefault(item, [0, 0])
    clusters = sorted(by_cluster)
    observed_num = sum(value[0] for value in by_cluster.values())
    observed_den = sum(value[1] for value in by_cluster.values())
    estimate = observed_num / observed_den if observed_den else None
    samples: list[float] = []
    if clusters and observed_den:
        rng = random.Random(SEED)
        for _ in range(BOOTSTRAPS):
            chosen = rng.choices(clusters, k=len(clusters))
            sample_num = sum(by_cluster[item][0] for item in chosen)
            sample_den = sum(by_cluster[item][1] for item in chosen)
            if sample_den:
                samples.append(sample_num / sample_den)
    return {
        "numerator": observed_num,
        "denominator": observed_den,
        "estimate": estimate,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)] if samples else [None, None],
        "material_clusters": len(clusters),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": SEED,
    }


def paired_difference_ci(
    outcomes: Sequence[dict[str, Any]], model: str, endpoint_materials: set[str]
) -> dict[str, Any]:
    cells: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"broad": [0, 0], "protocol": [0, 0]})
    for row in outcomes:
        if row["requested_model"] != model or row["material_id"] not in endpoint_materials:
            continue
        cell = cells[row["material_id"]][row["condition"]]
        cell[0] += int(row["correct_identity_recovered"])
        cell[1] += 1
    if set(cells) != endpoint_materials:
        raise ExtensionAnalysisError(f"paired endpoint material mismatch for {model}")
    for material, arms in cells.items():
        if arms["broad"][1] != arms["protocol"][1] or arms["broad"][1] == 0:
            raise ExtensionAnalysisError(f"paired denominator mismatch for {model}/{material}: {arms}")
    broad_num = sum(arms["broad"][0] for arms in cells.values())
    broad_den = sum(arms["broad"][1] for arms in cells.values())
    protocol_num = sum(arms["protocol"][0] for arms in cells.values())
    protocol_den = sum(arms["protocol"][1] for arms in cells.values())
    point = protocol_num / protocol_den - broad_num / broad_den
    materials = sorted(cells)
    rng = random.Random(SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.choices(materials, k=len(materials))
        b_num = sum(cells[item]["broad"][0] for item in chosen)
        b_den = sum(cells[item]["broad"][1] for item in chosen)
        p_num = sum(cells[item]["protocol"][0] for item in chosen)
        p_den = sum(cells[item]["protocol"][1] for item in chosen)
        samples.append(p_num / p_den - b_num / b_den)
    return {
        "broad": {"numerator": broad_num, "denominator": broad_den, "recall": broad_num / broad_den},
        "protocol": {"numerator": protocol_num, "denominator": protocol_den, "recall": protocol_num / protocol_den},
        "protocol_minus_broad": point,
        "ci95": [quantile(samples, 0.025), quantile(samples, 0.975)],
        "material_clusters": len(materials),
        "bootstrap_resamples": BOOTSTRAPS,
        "seed": SEED,
    }


def make_estimates(
    full_metrics: Sequence[dict[str, Any]],
    broad_outcomes: Sequence[dict[str, Any]],
    paired_outcomes: Sequence[dict[str, Any]],
    endpoint_materials: set[str],
    parse_counts: Counter[str],
    unresolved_count: int,
) -> dict[str, Any]:
    estimates: dict[str, Any] = {
        "analysis_version": "subscription-cli-extension-1.0",
        "collection": {"planned": 330, "completed": 330, "failed": 0, "unattempted": 0},
        "configurations": {
            "gpt-5.6-sol": {
                "requested_configuration": "gpt-5.6-sol",
                "returned_model_identifier": None,
                "returned_identifier_evidence": "not exposed by the Codex CLI stream",
                "client": "Codex CLI 0.157.1 via subscription",
                "effort": "high",
            },
            "claude-opus-5-5": {
                "requested_configuration": "claude-opus-5-5",
                "returned_model_identifier": "claude-opus-5-5",
                "returned_identifier_evidence": "exposed by the Claude CLI stream",
                "client": "Claude Code 2.1.283 via existing subscription",
                "effort": "high",
            },
        },
        "parse_status_counts": dict(parse_counts),
        "unresolved_conservative_partial_or_other_events": unresolved_count,
        "broad_identity_recall": {},
        "paired_protocol_nonmelting_recall": {},
        "overall_finite_reference_precision": {},
    }
    for model in MODELS:
        model_outcomes = [row for row in broad_outcomes if row["requested_model"] == model]
        broad_materials = {
            row["material_id"] for row in full_metrics
            if row["requested_model"] == model and row["condition"] == "broad"
        }
        estimates["broad_identity_recall"][model] = {}
        for event_class, label in (("terminal_melting", "terminal_melting"), ("non_melting_pathway", "non_melting")):
            rows = [row for row in model_outcomes if row["event_class"] == event_class]
            estimates["broad_identity_recall"][model][label] = aggregate_ratio_ci(
                [{**row, "den": 1} for row in rows], "correct_identity_recovered", "den",
                cluster_universe=broad_materials,
            )
        estimates["paired_protocol_nonmelting_recall"][model] = paired_difference_ci(
            paired_outcomes, model, endpoint_materials
        )
        estimates["overall_finite_reference_precision"][model] = {}
        for condition in ("broad", "protocol"):
            rows = [row for row in full_metrics if row["requested_model"] == model and row["condition"] == condition]
            estimates["overall_finite_reference_precision"][model][condition] = aggregate_ratio_ci(
                rows, "correct_identities", "predicted_events"
            )
    return estimates


def results_markdown(estimates: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{100 * value:.1f}%"

    lines = [
        "# Subscription-CLI extension results",
        "",
        "All 330 planned responses completed and parsed without repair: 180 broad and 150 protocol-specified responses, evenly divided between the two requested configurations. The original API benchmark remains unchanged and is not pooled with these results.",
        "",
        "## Broad-prompt identity recall",
        "",
    ]
    for model in MODELS:
        result = estimates["broad_identity_recall"][model]
        terminal = result["terminal_melting"]
        nonmelting = result["non_melting"]
        lines.append(
            f"- `{model}` recovered {terminal['numerator']}/{terminal['denominator']} terminal-melting units "
            f"({pct(terminal['estimate'])}; 95% material-cluster CI {pct(terminal['ci95'][0])} to {pct(terminal['ci95'][1])}) and "
            f"{nonmelting['numerator']}/{nonmelting['denominator']} non-melting units "
            f"({pct(nonmelting['estimate'])}; 95% CI {pct(nonmelting['ci95'][0])} to {pct(nonmelting['ci95'][1])})."
        )
    lines += ["", "## Protocol-context comparison", ""]
    for model in MODELS:
        result = estimates["paired_protocol_nonmelting_recall"][model]
        lines.append(
            f"- `{model}`: broad {result['broad']['numerator']}/{result['broad']['denominator']} "
            f"({pct(result['broad']['recall'])}); protocol {result['protocol']['numerator']}/{result['protocol']['denominator']} "
            f"({pct(result['protocol']['recall'])}); protocol-minus-broad {100 * result['protocol_minus_broad']:+.1f} percentage points "
            f"(95% paired material-cluster CI {100 * result['ci95'][0]:+.1f} to {100 * result['ci95'][1]:+.1f}; 22 materials)."
        )
    lines += [
        "",
        "The protocol contrast estimates the effect of the bundled preparation and scan context on recovery of the prespecified common non-melting units. It does not identify an internal knowledge mechanism. The broad and protocol arms use identical within-material reference denominators; all three generations travel together in each of 10,000 paired material-level bootstrap resamples (seed 20260927).",
        "",
        "## Finite-reference precision",
        "",
    ]
    for model in MODELS:
        pieces = []
        for condition in ("broad", "protocol"):
            result = estimates["overall_finite_reference_precision"][model][condition]
            pieces.append(
                f"{condition} {result['numerator']}/{result['denominator']} ({pct(result['estimate'])}; 95% CI {pct(result['ci95'][0])} to {pct(result['ci95'][1])})"
            )
        lines.append(f"- `{model}`: " + "; ".join(pieces) + ".")
    lines += [
        "",
        "Precision is finite-reference agreement: an emitted identity counts as correct only when it matches one of the source-qualified units. Protocol-specific omissions or extra events can therefore reflect genuine protocol mismatch as well as reconstruction error; the same mismatch can lower recall when a reference unit is not expected under the supplied branch.",
        "",
        "## Completion and scoring notes",
        "",
        f"All outputs were schema-valid. {estimates['unresolved_conservative_partial_or_other_events']} partial/ambiguous or `other` events remained conservatively non-correct without fabricated adjudicator consensus. Opus exposed the exact returned identifier `claude-opus-5-5`; the Codex CLI did not expose a returned identifier for the requested `gpt-5.6-sol` configuration, so none is inferred.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    INTERNAL.mkdir(parents=True, exist_ok=True)
    ledger, planned, packet = load_and_validate_collection()
    materials = load_json(ROOT / "cohort/materials.json")
    literature = load_json(ROOT / "cohort/literature_reference.json")
    materials_by_id = {row["material_id"]: row for row in materials["materials"]}
    molecular_units = [
        unit for unit in make_gold_units(literature, materials_by_id)
        if unit["stratum"] == "molecular_polymorph" and unit["counts_toward_primary"]
    ]
    condition_by_key = {row["key"]: row["condition"] for row in ledger}

    parsed, event_rows, adjudication_rows, _ = parse_and_match(ledger, molecular_units, decisions={})
    add_condition(parsed, condition_by_key)
    add_condition(event_rows, condition_by_key)
    attach_conditional_scores(event_rows, molecular_units)
    full_metrics = create_response_metrics(parsed, event_rows, molecular_units, materials_by_id)
    add_condition(full_metrics, condition_by_key)
    full_outcomes = make_gold_outcomes(full_metrics, event_rows, molecular_units)
    add_condition(full_outcomes, condition_by_key)

    eligibility = eligibility_rows()
    fixed_units, fixed_nonmelting, endpoint_materials = selected_units(eligibility)
    paired_units = [unit for unit in molecular_units if unit["scoring_unit_id"] in fixed_units]
    paired_ledger = [row for row in ledger if row["material_id"] in {x["material_id"] for x in eligibility if x["protocol_eligible"] == "yes"}]
    paired_parsed, paired_event_rows, paired_adjudications, _ = parse_and_match(paired_ledger, paired_units, decisions={})
    add_condition(paired_parsed, condition_by_key)
    add_condition(paired_event_rows, condition_by_key)
    paired_metrics = create_response_metrics(paired_parsed, paired_event_rows, paired_units, materials_by_id)
    add_condition(paired_metrics, condition_by_key)
    paired_outcomes = make_gold_outcomes(paired_metrics, paired_event_rows, paired_units)
    add_condition(paired_outcomes, condition_by_key)
    paired_nonmelting_outcomes = [row for row in paired_outcomes if row["scoring_unit_id"] in fixed_nonmelting]

    broad_outcomes = [row for row in full_outcomes if row["condition"] == "broad"]
    parse_counts = Counter(row["parse_status"] for row in parsed)
    unresolved_count = sum(row["adjudicated_label"] != "correct_identity" for row in adjudication_rows)
    estimates = make_estimates(
        full_metrics,
        broad_outcomes,
        paired_nonmelting_outcomes,
        endpoint_materials,
        parse_counts,
        unresolved_count,
    )

    write_csv(RESULTS / "response_metrics.csv", full_metrics)
    write_csv(RESULTS / "gold_outcomes.csv", full_outcomes)
    write_csv(RESULTS / "event_matches.csv", event_rows)
    write_csv(RESULTS / "paired_gold_outcomes.csv", paired_nonmelting_outcomes)
    write_csv(INTERNAL / "unresolved_adjudication.csv", adjudication_rows)
    write_csv(INTERNAL / "paired_unresolved_adjudication.csv", paired_adjudications)
    (RESULTS / "estimates.json").write_text(json.dumps(estimates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    accounting = {
        "planned": len(planned),
        "outcomes": len(ledger),
        "status_counts": {"completed": len(ledger)},
        "model_counts": dict(Counter(row["requested_model"] for row in ledger)),
        "condition_counts": dict(Counter(row["condition"] for row in ledger)),
        "parse_status_counts": dict(parse_counts),
        "broad_materials": len({row["material_id"] for row in ledger if row["condition"] == "broad"}),
        "protocol_materials": len({row["material_id"] for row in ledger if row["condition"] == "protocol"}),
        "paired_nonmelting_materials": len(endpoint_materials),
        "paired_fixed_units": len(fixed_units),
        "paired_nonmelting_units": len(fixed_nonmelting),
        "original_protected_hashes_reverified": len(load_json(EXT / "READY_FOR_PREFLIGHT.json")["protected_original_evidence_sha256"]),
        "outcomes_sha256": sha256(COLLECTION / "outcomes.jsonl"),
        "analysis_inputs": {
            "prompt_packet_sha256": sha256(EXT / "prompt_packet.json"),
            "eligibility_sha256": sha256(EXT / "eligibility.csv"),
            "literature_reference_sha256": sha256(ROOT / "cohort/literature_reference.json"),
        },
    }
    (RESULTS / "accounting.json").write_text(json.dumps(accounting, indent=2) + "\n", encoding="utf-8")
    (EXT / "RESULTS.md").write_text(results_markdown(estimates), encoding="utf-8")


if __name__ == "__main__":
    main()
