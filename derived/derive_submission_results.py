#!/usr/bin/env python3
"""Derive submission-only descriptive summaries from protected scored rows.

This script reads the existing original and extension scoring tables without
modifying them. Outputs are post hoc descriptive summaries used in the
2026-09-27 submission packages; they do not alter the frozen benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
OUT = HERE.parent

ORIGINAL_GOLD = ROOT / "data/original/gold_outcomes.csv"
ORIGINAL_MATCHES = ROOT / "data/original/event_matches.csv"
EXTENSION_GOLD = ROOT / "data/extension/gold_outcomes.csv"
EXTENSION_MATCHES = ROOT / "data/extension/event_matches.csv"
EXTENSION_ESTIMATES = ROOT / "data/extension/estimates.json"
REFERENCE = ROOT / "study/literature_reference.json"

MODELS = ["gpt-5.6-luna", "gpt-6-luna", "gpt-5.6-sol", "claude-opus-5-5"]
MODEL_LABELS = {
    "gpt-5.6-luna": "Luna 5.6 API",
    "gpt-6-luna": "Luna 6 API",
    "gpt-5.6-sol": "Sol CLI",
    "claude-opus-5-5": "Opus CLI",
}
FAMILIES = [
    "melting",
    "solid_solid_transition",
    "glass_transition",
    "cold_crystallization",
    "crystallization_on_cooling",
    "melt_mediated_conversion",
    "melt_sublimation_recrystallization",
]
FAMILY_LABELS = {
    "melting": "Terminal melting",
    "solid_solid_transition": "Solid-solid transition",
    "glass_transition": "Glass transition",
    "cold_crystallization": "Cold crystallization",
    "crystallization_on_cooling": "Cooling crystallization",
    "melt_mediated_conversion": "Melt-mediated conversion",
    "melt_sublimation_recrystallization": "Melt/sublimation/recrystallization",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_value(value: str) -> bool | None:
    value = str(value).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    return None


def family_summary() -> list[dict[str, object]]:
    original = [
        row for row in rows(ORIGINAL_GOLD)
        if row["stratum"] == "molecular_polymorph"
    ]
    extension = [
        row for row in rows(EXTENSION_GOLD)
        if row["stratum"] == "molecular_polymorph" and row["condition"] == "broad"
    ]
    combined = original + extension
    output: list[dict[str, object]] = []
    for model in MODELS:
        model_rows = [row for row in combined if row["requested_model"] == model]
        for family in FAMILIES:
            subset = [row for row in model_rows if row["event_type"] == family]
            numerator = sum(int(row["correct_identity_recovered"]) for row in subset)
            denominator = len(subset)
            output.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "event_family": family,
                "event_family_label": FAMILY_LABELS[family],
                "numerator": numerator,
                "denominator": denominator,
                "recall": numerator / denominator if denominator else None,
                "scope": "broad molecular prompts; three generations; repeated opportunities",
            })
    return output


def conditional_summary() -> list[dict[str, object]]:
    original = [
        row for row in rows(ORIGINAL_MATCHES)
        if row["row_type"] == "predicted_event"
        and row["final_label"] == "correct_identity"
        and row["material_id"].startswith("mol_")
    ]
    extension = [
        row for row in rows(EXTENSION_MATCHES)
        if row["row_type"] == "predicted_event"
        and row["final_label"] == "correct_identity"
        and row["condition"] == "broad"
        and row["material_id"].startswith("mol_")
    ]
    combined = original + extension
    output: list[dict[str, object]] = []
    for model in MODELS:
        model_rows = [row for row in combined if row["requested_model"] == model]
        for event_class in ("terminal_melting", "non_melting_pathway"):
            subset = [row for row in model_rows if row["gold_event_class"] == event_class]
            direction_rows = [row for row in subset if bool_value(row["direction_correct"]) is not None]
            direction_correct = sum(bool_value(row["direction_correct"]) is True for row in direction_rows)
            temp_rows = [row for row in subset if row["temperature_absolute_error_C"] != ""]
            errors = [float(row["temperature_absolute_error_C"]) for row in temp_rows]
            output.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "event_class": event_class,
                "identity_correct_n": len(subset),
                "direction_scorable_n": len(direction_rows),
                "direction_correct_n": direction_correct,
                "direction_accuracy": direction_correct / len(direction_rows) if direction_rows else None,
                "direction_coverage": len(direction_rows) / len(subset) if subset else None,
                "temperature_compatible_n": len(temp_rows),
                "temperature_coverage": len(temp_rows) / len(subset) if subset else None,
                "temperature_material_n": len({row["material_id"] for row in temp_rows}),
                "temperature_median_absolute_error_C": statistics.median(errors) if errors else None,
                "temperature_max_absolute_error_C": max(errors) if errors else None,
                "scope": "identity-correct broad-prompt matches only",
            })
    return output


def protocol_summary() -> list[dict[str, object]]:
    with EXTENSION_ESTIMATES.open(encoding="utf-8") as stream:
        estimates = json.load(stream)
    output = []
    for model, values in estimates["paired_protocol_nonmelting_recall"].items():
        output.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "broad_numerator": values["broad"]["numerator"],
            "protocol_numerator": values["protocol"]["numerator"],
            "denominator_each": values["broad"]["denominator"],
            "broad_recall": values["broad"]["recall"],
            "protocol_recall": values["protocol"]["recall"],
            "difference": values["protocol_minus_broad"],
            "ci95_low": values["ci95"][0],
            "ci95_high": values["ci95"][1],
            "material_clusters": values["material_clusters"],
            "fixed_units": 41,
            "generations": 3,
        })
    return output


def illustrative_pathways() -> list[dict[str, object]]:
    selected = {"mol_paracetamol", "mol_carbamazepine", "mol_flufenamic_acid"}
    original = [
        row for row in rows(ORIGINAL_GOLD)
        if row["stratum"] == "molecular_polymorph" and row["material_id"] in selected
    ]
    extension = [
        row for row in rows(EXTENSION_GOLD)
        if row["stratum"] == "molecular_polymorph"
        and row["condition"] == "broad"
        and row["material_id"] in selected
    ]
    with REFERENCE.open(encoding="utf-8") as stream:
        reference = json.load(stream)
    anchors_by_unit: dict[str, list[dict[str, object]]] = defaultdict(list)
    for anchor in reference["anchors"]:
        if anchor["material_id"] in selected and anchor.get("counts_toward_primary", True):
            anchors_by_unit[anchor["scoring_unit_id"]].append(anchor)
    output = []
    for row in original + extension:
        unit = row["scoring_unit_id"]
        anchors = anchors_by_unit[unit]
        anchor = anchors[0]
        temperature = anchor["temperature"]
        if temperature.get("lower_C") is not None:
            temperature_text = f"{temperature['lower_C']:g}-{temperature['upper_C']:g} °C ({temperature['kind']})"
        elif temperature.get("value_C") is not None:
            temperature_text = f"{temperature['value_C']:g} °C ({temperature['kind']})"
        else:
            temperature_text = f"{temperature['kind']}"
        output.append({
            "material_id": row["material_id"],
            "scoring_unit_id": unit,
            "event_identity": " | ".join(str(a["event_identity"]) for a in anchors),
            "event_type": row["event_type"],
            "source_temperature": temperature_text,
            "source_locator": str(anchor["citation"]["locator"]),
            "model": row["requested_model"],
            "model_label": MODEL_LABELS[row["requested_model"]],
            "generation": int(row["generation"]),
            "recovered": int(row["correct_identity_recovered"]),
        })
    aggregate: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in output:
        key = (str(row["material_id"]), str(row["scoring_unit_id"]), str(row["model"]))
        if key not in aggregate:
            aggregate[key] = {**row, "recovered_generations": 0, "generation_opportunities": 0}
        aggregate[key]["recovered_generations"] = int(aggregate[key]["recovered_generations"]) + int(row["recovered"])
        aggregate[key]["generation_opportunities"] = int(aggregate[key]["generation_opportunities"]) + 1
    data = []
    for row in aggregate.values():
        row.pop("generation")
        row.pop("recovered")
        data.append(row)
    return sorted(data, key=lambda row: (row["material_id"], row["scoring_unit_id"], MODELS.index(str(row["model"]))))


def write_csv(path: Path, data: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    families = family_summary()
    conditional = conditional_summary()
    protocol = protocol_summary()
    examples = illustrative_pathways()
    write_csv(OUT / "event_family_recall.csv", families)
    write_csv(OUT / "conditional_direction_temperature.csv", conditional)
    write_csv(OUT / "paired_protocol_recall.csv", protocol)
    write_csv(OUT / "illustrative_pathways.csv", examples)

    expected_family = {
        "gpt-5.6-luna": [93, 43, 8, 13, 1, 0, 0],
        "gpt-6-luna": [101, 26, 4, 6, 2, 2, 0],
        "gpt-5.6-sol": [97, 35, 19, 4, 0, 10, 0],
        "claude-opus-5-5": [91, 64, 24, 7, 1, 6, 0],
    }
    expected_denominators = [108, 90, 24, 21, 6, 15, 3]
    checks = []
    for model in MODELS:
        subset = [row for row in families if row["model"] == model]
        observed_n = [row["numerator"] for row in subset]
        observed_d = [row["denominator"] for row in subset]
        checks.append({
            "name": f"event_family_counts::{model}",
            "pass": observed_n == expected_family[model] and observed_d == expected_denominators,
            "observed_numerators": observed_n,
            "observed_denominators": observed_d,
        })

    expected_conditional = {
        ("gpt-5.6-sol", "terminal_melting"): (97, 97, 7, 0.4, 14.0),
        ("gpt-5.6-sol", "non_melting_pathway"): (68, 65, 11, 0.0, 15.0),
        ("claude-opus-5-5", "terminal_melting"): (91, 91, 58, 1.0, 24.42),
        ("claude-opus-5-5", "non_melting_pathway"): (102, 78, 22, 0.0, 8.1),
    }
    for key, expected in expected_conditional.items():
        row = next(item for item in conditional if (item["model"], item["event_class"]) == key)
        observed = (
            row["identity_correct_n"], row["direction_scorable_n"], row["temperature_compatible_n"],
            row["temperature_median_absolute_error_C"], row["temperature_max_absolute_error_C"],
        )
        numeric_match = (
            observed[:3] == expected[:3]
            and abs(float(observed[3]) - expected[3]) < 1e-9
            and abs(float(observed[4]) - expected[4]) < 1e-9
        )
        checks.append({
            "name": f"conditional_coverage::{key[0]}::{key[1]}",
            "pass": numeric_match,
            "observed": observed,
            "expected": expected,
        })

    verification = {
        "status": "PASS" if all(check["pass"] for check in checks) else "FAIL",
        "description": "Independent row-level aggregation for submission-only descriptive summaries.",
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (
            ORIGINAL_GOLD, ORIGINAL_MATCHES, EXTENSION_GOLD, EXTENSION_MATCHES,
            EXTENSION_ESTIMATES, REFERENCE
        )},
        "checks": checks,
    }
    (OUT / "verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    if verification["status"] != "PASS":
        raise SystemExit("submission derivation checks failed")


if __name__ == "__main__":
    main()
