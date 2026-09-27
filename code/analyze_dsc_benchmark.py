#!/usr/bin/env python3
"""Reproducible analysis for the frozen DSC API benchmark.

The raw collection is read-only.  This module validates the complete collection,
extracts completed structured outputs, performs temperature-blind identity
matching, applies a separately reviewed blinded-adjudication table, computes the
prespecified analyses, and writes only to the requested results directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2, norm
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE


STUDY_ID = "DSC-CHEMRXIV-COHORT-2026-09-26"
EXPECTED_MODELS = ("gpt-5.6-luna", "gpt-6-luna")
EXPECTED_REQUESTS = 222
EXPECTED_COMPLETED = 219
EXPECTED_INCOMPLETE = 3
BOOTSTRAP_SEED = 20260927
ANALYSIS_TIMESTAMP_UTC = "2026-09-27T00:00:00+00:00"
STRATA = ("molecular_polymorph", "polymer_control", "complex_material")
STRATUM_LABELS = {
    "molecular_polymorph": "Molecular polymorphs",
    "polymer_control": "Polymer controls",
    "complex_material": "Cocoa butter",
}
MODELS_SOL = {
    "gpt-5.6-sol": {"input_per_million_usd": 4.0, "output_per_million_usd": 20.0},
    "gpt-6-sol": {"input_per_million_usd": 2.0, "output_per_million_usd": 10.0},
}
MODELS_LUNA = {
    "gpt-5.6-luna": {"input_per_million_usd": 0.20, "output_per_million_usd": 1.20},
    "gpt-6-luna": {"input_per_million_usd": 0.10, "output_per_million_usd": 0.50},
}
OFFICIAL_DOCS = {
    "gpt-5.6-sol": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    "gpt-6-sol": "https://developers.openai.com/api/docs/models/gpt-6-sol",
    "pricing": "https://developers.openai.com/api/docs/pricing",
    "reasoning": "https://developers.openai.com/api/docs/guides/reasoning",
    "token_counting": "https://developers.openai.com/api/docs/guides/token-counting",
}


class AnalysisError(RuntimeError):
    """Fail-closed validation or analysis error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise AnalysisError(f"Non-object JSONL record in {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    answer = float(value)
    return answer if math.isfinite(answer) else None


def quantile(values: Sequence[float], probability: float) -> float | None:
    clean = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))])
    return None if clean.size == 0 else float(np.quantile(clean, probability))


def describe(values: Sequence[float]) -> dict[str, Any]:
    clean = np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))])
    if clean.size == 0:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None, "sum": 0.0}
    return {
        "n": int(clean.size),
        "min": float(np.min(clean)),
        "p25": float(np.quantile(clean, 0.25)),
        "median": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "p90": float(np.quantile(clean, 0.90)),
        "p95": float(np.quantile(clean, 0.95)),
        "p99": float(np.quantile(clean, 0.99)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "sum": float(np.sum(clean)),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (n - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def validate_collection(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    run_dir = root / "outputs" / "luna-api-run-2026-09-27"
    paths = {
        "manifest": run_dir / "MANIFEST.json",
        "plan": run_dir / "execution-plan.jsonl",
        "attempts": run_dir / "attempt-ledger.jsonl",
        "responses": run_dir / "response-ledger.jsonl",
        "gold": root / "cohort" / "literature_reference.json",
        "materials": root / "cohort" / "materials.json",
        "packet": root / "blind" / "MODEL_PACKET.json",
    }
    before_hashes = {name: sha256_file(path) for name, path in paths.items()}
    manifest = load_json(paths["manifest"])
    plan = load_jsonl(paths["plan"])
    attempts = load_jsonl(paths["attempts"])
    responses = load_jsonl(paths["responses"])
    packet = load_json(paths["packet"])

    failures: list[str] = []
    if manifest.get("study_id") != STUDY_ID or packet.get("study_id") != STUDY_ID:
        failures.append("study_id mismatch")
    if manifest.get("status") != "COMPLETE":
        failures.append("collection manifest is not COMPLETE")
    if len(plan) != EXPECTED_REQUESTS or len(attempts) != EXPECTED_REQUESTS or len(responses) != EXPECTED_REQUESTS:
        failures.append(f"ledger totals are not all {EXPECTED_REQUESTS}")
    key_sets = [set(row.get("key") for row in rows) for rows in (plan, attempts, responses)]
    if any(len(keys) != EXPECTED_REQUESTS for keys in key_sets) or not (key_sets[0] == key_sets[1] == key_sets[2]):
        failures.append("planned/attempt/response keys are not 222 unique identical keys")

    plan_by_key = {row["key"]: row for row in plan}
    attempt_by_key = {row["key"]: row for row in attempts}
    packet_by_material = {row["material_id"]: row for row in packet.get("packets", [])}
    cells: Counter[tuple[str, str]] = Counter()
    state_counts: Counter[str] = Counter()
    response_ids: set[str] = set()
    raw_response_json_valid = 0
    required = set(manifest.get("frozen_api_plan", {}).get("required_response_ledger_fields", []))
    for row in responses:
        key = row.get("key")
        planned = plan_by_key.get(key, {})
        attempted = attempt_by_key.get(key, {})
        missing = required - set(row)
        if missing:
            failures.append(f"{key}: missing required fields {sorted(missing)}")
        prompt_hash = sha256_bytes(str(row.get("prompt_text", "")).encode("utf-8"))
        if prompt_hash != row.get("prompt_sha256") or prompt_hash != planned.get("prompt_sha256"):
            failures.append(f"{key}: prompt hash mismatch")
        packet_row = packet_by_material.get(row.get("material_id"), {})
        if row.get("prompt_sha256") != packet_row.get("prompt_sha256"):
            failures.append(f"{key}: packet prompt hash mismatch")
        for field in ("material_id", "requested_model", "generation", "prompt_text"):
            if row.get(field) != planned.get(field):
                failures.append(f"{key}: plan/response {field} mismatch")
        requested = row.get("requested_model")
        returned = row.get("returned_model")
        if requested not in EXPECTED_MODELS or returned != requested:
            failures.append(f"{key}: requested/returned model mismatch {requested!r}/{returned!r}")
        cells[(str(row.get("material_id")), str(requested))] += 1
        if row.get("generation") not in (1, 2, 3):
            failures.append(f"{key}: generation outside 1..3")
        state_counts[str(row.get("completion_state"))] += 1
        response_id = row.get("response_id")
        if not response_id or response_id in response_ids:
            failures.append(f"{key}: missing or duplicate response_id")
        response_ids.add(response_id)
        if attempted.get("response_id") != response_id or attempted.get("completion_state") != row.get("completion_state"):
            failures.append(f"{key}: attempt/response identity or status mismatch")
        if row.get("attempt_number") != 1 or attempted.get("attempt_number") != 1:
            failures.append(f"{key}: unexpected retry/attempt number")
        try:
            raw = json.loads(row.get("raw_response_json", ""))
        except (TypeError, json.JSONDecodeError):
            failures.append(f"{key}: invalid embedded raw_response_json")
        else:
            raw_response_json_valid += 1
            checks = {
                "id": response_id,
                "model": returned,
                "status": row.get("completion_state"),
                "usage": row.get("usage"),
                "output": row.get("raw_output"),
            }
            for field, expected in checks.items():
                if raw.get(field) != expected:
                    failures.append(f"{key}: raw response {field} disagrees with ledger")

    materials = {row.get("material_id") for row in load_json(paths["materials"]).get("materials", [])}
    if len(materials) != 37 or len(packet_by_material) != 37:
        failures.append("materials/model packet do not each contain 37 unique materials")
    expected_cells = {(material, model) for material in materials for model in EXPECTED_MODELS}
    if set(cells) != expected_cells or any(value != 3 for value in cells.values()):
        failures.append("not every model-material cell has exactly three generations")
    if state_counts != Counter({"completed": EXPECTED_COMPLETED, "incomplete": EXPECTED_INCOMPLETE}):
        failures.append(f"unexpected completion totals: {dict(state_counts)}")
    if manifest.get("completed_terminal_records") != EXPECTED_REQUESTS or manifest.get("completion_state_counts") != dict(state_counts):
        failures.append("manifest totals disagree with derived totals")
    if manifest.get("planned_requests") != EXPECTED_REQUESTS or manifest.get("materials") != 37:
        failures.append("manifest planned totals disagree with frozen design")
    if raw_response_json_valid != EXPECTED_REQUESTS:
        failures.append("not all embedded raw responses validated")
    if failures:
        preview = "\n".join(f"- {item}" for item in failures[:50])
        raise AnalysisError(f"Collection validation failed ({len(failures)} defects):\n{preview}")
    return manifest, plan, attempts, responses, before_hashes


def extract_output_text(raw_output: Any) -> str:
    if not isinstance(raw_output, list):
        return ""
    pieces: list[str] = []
    for item in raw_output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                pieces.append(part["text"])
    return "".join(pieces)


REQUIRED_EVENT_FIELDS = {
    "sequence", "starting_form_state", "product_form_state", "event_type",
    "direction", "temperature", "confidence", "note",
}
EVENT_FAMILIES = {
    "glass_transition", "crystallization_on_cooling", "cold_crystallization",
    "solid_solid_transition", "melt_mediated_conversion",
    "melt_sublimation_recrystallization", "melting", "recrystallization", "other",
}


def validate_structured_output(value: Any, expected_material: str) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "top_level_not_object"
    # The frozen schema constrained this field only to string, not to the
    # machine material key.  Models consistently emitted the human material
    # name.  The response is already bound to a validated immutable request
    # key, so a nonempty string is retained verbatim and the ledger key remains
    # authoritative for joining; no raw value is rewritten.
    if not isinstance(value.get("material_id"), str) or not value.get("material_id", "").strip():
        return False, "material_id_missing"
    if value.get("schematic_reconstruction") is not True:
        return False, "schematic_flag_invalid"
    if not isinstance(value.get("protocol_assumptions"), list) or not isinstance(value.get("events"), list):
        return False, "top_level_schema_invalid"
    for index, event in enumerate(value["events"]):
        if not isinstance(event, dict) or set(event) != REQUIRED_EVENT_FIELDS:
            return False, f"event_{index}_fields_invalid"
        if event.get("event_type") not in EVENT_FAMILIES:
            return False, f"event_{index}_family_invalid"
        if not isinstance(event.get("temperature"), dict):
            return False, f"event_{index}_temperature_invalid"
        if not isinstance(event.get("confidence"), (int, float)) or not 0 <= float(event["confidence"]) <= 1:
            return False, f"event_{index}_confidence_invalid"
    return True, "valid"


@dataclass(frozen=True)
class State:
    raw: str
    normalized: str
    forms: frozenset[str]
    categories: frozenset[str]
    tokens: frozenset[str]
    generic: bool
    operations: tuple[str, ...]


GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "Α": "alpha", "Β": "beta", "Γ": "gamma", "Δ": "delta", "Ε": "epsilon",
}
ROMAN_UNICODE = {"Ⅰ": "i", "Ⅱ": "ii", "Ⅲ": "iii", "Ⅳ": "iv", "Ⅴ": "v", "Ⅵ": "vi", "Ⅸ": "ix", "Ⅹ": "x"}
STOP_TOKENS = {
    "state", "form", "material", "sample", "source", "specified", "starting", "converted",
    "precursor", "derived", "after", "before", "post", "heating", "second", "reference",
    "the", "a", "an", "of", "or", "and", "via", "to", "in", "under", "phase",
}


def normalize_state(raw: Any) -> State:
    original = "" if raw is None else str(raw)
    text = unicodedata.normalize("NFKC", original)
    operations: list[str] = []
    translated = "".join(GREEK.get(char, ROMAN_UNICODE.get(char, char)) for char in text)
    if translated != text:
        operations.append("unicode_greek_or_roman")
    text = translated.lower().replace("–", "-").replace("—", "-").replace("→", " to ")
    replacements = {
        "polymorph": "form", "melt quenched": "melt-quenched", "super cooled": "supercooled",
        "poly(methyl methacrylate)": "pmma", "polymethyl methacrylate": "pmma",
        "polystyrene": "ps", "polyethylene terephthalate": "pet",
        "poly(l-lactic acid)": "plla", "isotactic polypropylene": "ipp",
        "polyvinylidene fluoride": "pvdf", "molten": "liquid", "melted": "liquid",
        "vitreous": "amorphous", "vitrified": "amorphous", "rubber-like": "rubbery",
        "sub-alpha": "sub alpha",
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            operations.append(f"synonym:{old}->{new}")
    forms: set[str] = set()
    for match in re.finditer(r"\bform\s*[- ]?([ivx]+|[abc])\b", text):
        forms.add(f"form_{match.group(1)}")
    for match in re.finditer(r"\bforms?\s+([ivx]+(?:\s*[,/&-]|\s+and\s+|\s+or\s+)[ivx]+(?:\s*[,/&-]|\s+and\s+|\s+or\s+|[ivx])*)", text):
        for roman in re.findall(r"\b(?:i{1,3}|iv|v|vi|vii|viii|ix|x)\b", match.group(1)):
            forms.add(f"form_{roman}")
    if "pattern x" in text:
        forms.add("form_x")
        operations.append("pattern_x->form_x")
    for greek in ("alpha", "beta", "gamma", "delta", "epsilon"):
        if re.search(rf"\b{greek}\b", text):
            forms.add(f"form_{greek}")

    categories: set[str] = set()
    melt_quenched = "melt-quenched" in text
    if re.search(r"\bliquid\b", text) or (re.search(r"\bmelt\b", text) and not melt_quenched):
        categories.add("liquid")
    if any(word in text for word in ("amorphous", "glass", "melt-quenched", "supercooled state")):
        categories.add("amorphous")
    if "supercooled liquid" in text or "rubbery" in text:
        categories.add("rubbery")
    if any(word in text for word in ("crystal", "crystalline", "semicrystalline")) or forms:
        categories.add("crystalline")
    for polymer in ("ps", "pmma", "pet", "plla", "ipp", "pvdf"):
        if re.search(rf"\b{polymer}\b", text):
            categories.add(polymer)
    if "fat" in text or "cocoa butter" in text:
        categories.add("cocoa_butter")
    generic = any(phrase in text for phrase in (
        "not reported", "unspecified", "unknown", "generic", "or converted precursor",
        "converted precursor", "corresponding", "one or more", "various", "mixed forms",
        "crystalline state", "crystalline material", "a crystalline", "stable form",
    ))
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    tokens = frozenset(token for token in normalized.split() if token not in STOP_TOKENS and len(token) > 1)
    if normalized != re.sub(r"\s+", " ", original.lower()).strip():
        operations.append("case_punctuation_whitespace")
    return State(original, normalized, frozenset(forms), frozenset(categories), tokens, generic, tuple(dict.fromkeys(operations)))


def state_relation(gold_raw: Any, predicted_raw: Any) -> tuple[str, str, State, State]:
    gold = normalize_state(gold_raw)
    pred = normalize_state(predicted_raw)
    if gold.normalized == pred.normalized and gold.normalized:
        return "exact", "normalized strings agree", gold, pred
    if gold.forms and pred.forms:
        if gold.forms & pred.forms:
            return "exact", f"canonical form overlap {sorted(gold.forms & pred.forms)}", gold, pred
        return "conflict", f"canonical forms conflict {sorted(gold.forms)} vs {sorted(pred.forms)}", gold, pred
    if "liquid" in gold.categories and "liquid" in pred.categories:
        return "exact", "liquid/melt endpoint synonym", gold, pred
    if "amorphous" in gold.categories and "amorphous" in pred.categories:
        return "exact", "amorphous/vitrified state synonym", gold, pred
    if "rubbery" in gold.categories and ("rubbery" in pred.categories or "liquid" in pred.categories):
        return "exact", "rubbery/supercooled-liquid endpoint synonym", gold, pred
    if "rubbery" in pred.categories and ("rubbery" in gold.categories or "liquid" in gold.categories):
        return "exact", "rubbery/supercooled-liquid endpoint synonym", gold, pred
    common_polymer = (gold.categories & pred.categories) & {"ps", "pmma", "pet", "plla", "ipp", "pvdf"}
    if common_polymer:
        if ({"liquid", "amorphous", "rubbery", "crystalline"} & gold.categories) and ({"liquid", "amorphous", "rubbery", "crystalline"} & pred.categories):
            phase_gold = {"liquid", "amorphous", "rubbery", "crystalline"} & gold.categories
            phase_pred = {"liquid", "amorphous", "rubbery", "crystalline"} & pred.categories
            if phase_gold & phase_pred:
                return "exact", f"polymer and phase agree ({sorted(common_polymer)})", gold, pred
        return "partial", f"polymer agrees but phase is underspecified ({sorted(common_polymer)})", gold, pred
    union = gold.tokens | pred.tokens
    overlap = gold.tokens & pred.tokens
    if union and len(overlap) / len(union) >= 0.60:
        return "exact", f"state-token overlap {len(overlap)}/{len(union)}", gold, pred
    if gold.generic or pred.generic:
        return "partial", "one state is explicitly generic or unresolved", gold, pred
    if ("crystalline" in gold.categories and "crystalline" in pred.categories) or (not gold.tokens or not pred.tokens):
        return "partial", "crystalline/state identity is underspecified", gold, pred
    return "conflict", "named states do not agree", gold, pred


def make_gold_units(gold: dict[str, Any], materials_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in gold.get("anchors", []):
        grouped[anchor["scoring_unit_id"]].append(anchor)
    units: list[dict[str, Any]] = []
    for scoring_unit_id, anchors in grouped.items():
        first = anchors[0]
        material = materials_by_id[first["material_id"]]
        if any(anchor["material_id"] != first["material_id"] or anchor["event_type"] != first["event_type"] for anchor in anchors):
            raise AnalysisError(f"Inconsistent gold anchors grouped under {scoring_unit_id}")
        starts = list(dict.fromkeys(anchor["starting_form_state"] for anchor in anchors))
        products = list(dict.fromkeys(anchor["product_form_state"] for anchor in anchors))
        identities = list(dict.fromkeys(anchor["event_identity"] for anchor in anchors))
        units.append({
            "scoring_unit_id": scoring_unit_id,
            "anchor_ids": [anchor["anchor_id"] for anchor in anchors],
            "material_id": first["material_id"],
            "stratum": material["stratum"],
            "complexity_tier": material["complexity_tier"],
            "event_type": first["event_type"],
            "event_class": first["event_class"],
            "starting_form_states": starts,
            "product_form_states": products,
            "event_identities": identities,
            "directions": list(dict.fromkeys(anchor["direction"] for anchor in anchors)),
            "temperature_anchors": [anchor["temperature"] for anchor in anchors],
            "protocol_id": first["protocol_id"],
            "counts_toward_primary": bool(first["counts_toward_primary"]),
            "gold_definition": " | ".join(f"{anchor['event_type']}: {anchor['event_identity']}" for anchor in anchors),
        })
    units.sort(key=lambda row: (row["material_id"], row["scoring_unit_id"]))
    return units


def relation_to_gold(gold_unit: dict[str, Any], event: dict[str, Any]) -> tuple[str, str, list[str]]:
    family = event.get("event_type")
    if family == "other":
        return "other", "event family is other and requires adjudication", []
    if family != gold_unit["event_type"]:
        return "ineligible", "event families differ", []
    start_relations = [state_relation(gold_state, event.get("starting_form_state")) for gold_state in gold_unit["starting_form_states"]]
    product_relations = [state_relation(gold_state, event.get("product_form_state")) for gold_state in gold_unit["product_form_states"]]
    rank = {"exact": 2, "partial": 1, "conflict": 0}
    best_start = max(start_relations, key=lambda item: rank[item[0]])
    best_product = max(product_relations, key=lambda item: rank[item[0]])
    operations = list(best_start[3].operations) + list(best_product[3].operations)
    if best_start[0] == "conflict" or best_product[0] == "conflict":
        return "ineligible", f"family agrees; start={best_start[1]}; product={best_product[1]}", operations
    if best_start[0] == "exact" and best_product[0] == "exact":
        return "correct_identity", f"family agrees; start={best_start[1]}; product={best_product[1]}", operations
    return "partial_ambiguous", f"family agrees; start={best_start[1]}; product={best_product[1]}", operations


def predicted_identity_key(event: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(event.get("event_type", "")),
        normalize_state(event.get("starting_form_state")).normalized,
        normalize_state(event.get("product_form_state")).normalized,
    )


def predicted_identity_from_event_row(row: dict[str, Any]) -> str:
    """Canonical identity derived only from the model-produced event statement."""
    return "|".join((
        str(row.get("predicted_event_type", "")),
        normalize_state(row.get("predicted_starting_form_state")).normalized,
        normalize_state(row.get("predicted_product_form_state")).normalized,
    ))


def deduplicate_predicted_events(events: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str], int] = {}
    for event in events:
        key = predicted_identity_key(event)
        if key not in seen:
            copy = dict(event)
            copy["_original_indices"] = [event["_event_index"]]
            seen[key] = len(retained)
            retained.append(copy)
        else:
            kept = retained[seen[key]]
            kept["_original_indices"].append(event["_event_index"])
            duplicates.append({**event, "_duplicate_of": kept["_event_index"], "_deduplication_reason": "exact_canonical_identity_duplicate"})
    return retained, duplicates


def deduplicate_shared_terminal_melts(
    events: Sequence[dict[str, Any]], gold_units: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained = list(events)
    duplicates: list[dict[str, Any]] = []
    remove_indices: set[int] = set()
    common_units = [unit for unit in gold_units if unit["event_type"] == "melting" and "__common" in unit["scoring_unit_id"]]
    for unit in common_units:
        candidates: list[tuple[int, int]] = []
        for index, event in enumerate(retained):
            if index in remove_indices:
                continue
            label, _, _ = relation_to_gold(unit, event)
            if label in {"correct_identity", "partial_ambiguous"}:
                candidates.append((index, 2 if label == "correct_identity" else 1))
        if len(candidates) <= 1:
            continue
        keep_index = max(candidates, key=lambda item: (item[1], -int(retained[item[0]]["_event_index"])))[0]
        kept = retained[keep_index]
        for index, _ in candidates:
            if index == keep_index:
                continue
            duplicate = dict(retained[index])
            duplicate["_duplicate_of"] = kept["_event_index"]
            duplicate["_deduplication_reason"] = f"shared_terminal_melt:{unit['scoring_unit_id']}"
            duplicates.append(duplicate)
            kept["_original_indices"].extend(duplicate.get("_original_indices", [duplicate["_event_index"]]))
            remove_indices.add(index)
    return [event for index, event in enumerate(retained) if index not in remove_indices], duplicates


def maximum_cardinality_assignment(events: Sequence[dict[str, Any]], gold_units: Sequence[dict[str, Any]]) -> tuple[list[tuple[int, int, str, str, list[str]]], list[dict[str, Any]]]:
    edges_by_pred: list[list[tuple[int, str, str, list[str]]]] = []
    ambiguity_records: list[dict[str, Any]] = []
    for pred_index, event in enumerate(events):
        edges: list[tuple[int, str, str, list[str]]] = []
        for gold_index, unit in enumerate(gold_units):
            label, rationale, operations = relation_to_gold(unit, event)
            if label in {"correct_identity", "partial_ambiguous"}:
                edges.append((gold_index, label, rationale, operations))
        edges_by_pred.append(edges)
        if event.get("event_type") == "other" or not edges:
            same_family = [unit for unit in gold_units if unit["event_type"] == event.get("event_type")]
            ambiguity_records.append({"pred_index": pred_index, "reason": "other" if event.get("event_type") == "other" else "no_compatible_identity", "candidates": same_family})

    @lru_cache(maxsize=None)
    def solve(pred_index: int, used_mask: int) -> tuple[tuple[int, int, int], tuple[tuple[int, int, str, str, tuple[str, ...]], ...]]:
        if pred_index == len(events):
            return (0, 0, 0), ()
        best_score, best_assignment = solve(pred_index + 1, used_mask)
        for gold_index, label, rationale, operations in edges_by_pred[pred_index]:
            if used_mask & (1 << gold_index):
                continue
            tail_score, tail_assignment = solve(pred_index + 1, used_mask | (1 << gold_index))
            exact = 1 if label == "correct_identity" else 0
            partial = 1 if label == "partial_ambiguous" else 0
            candidate_score = (tail_score[0] + 1, tail_score[1] + exact, tail_score[2] - partial)
            candidate_assignment = ((pred_index, gold_index, label, rationale, tuple(operations)),) + tail_assignment
            if candidate_score > best_score:
                best_score, best_assignment = candidate_score, candidate_assignment
        return best_score, best_assignment

    _, assignment = solve(0, 0)
    return [(a, b, c, d, list(e)) for a, b, c, d, e in assignment], ambiguity_records


def blind_response_id(row: dict[str, Any]) -> str:
    payload = f"{STUDY_ID}|{row['response_id']}|{row['material_id']}|{row['generation']}".encode("utf-8")
    return "blind_" + hashlib.sha256(payload).hexdigest()[:16]


def adjudication_id(blind_id: str, event_index: int, gold_unit_id: str | None, reason: str) -> str:
    payload = f"{blind_id}|{event_index}|{gold_unit_id or ''}|{reason}".encode("utf-8")
    return "adj_" + hashlib.sha256(payload).hexdigest()[:18]


def parse_and_match(
    responses: Sequence[dict[str, Any]],
    gold_units: Sequence[dict[str, Any]],
    decisions: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    decisions = decisions or {}
    gold_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in gold_units:
        gold_by_material[unit["material_id"]].append(unit)
    event_rows: list[dict[str, Any]] = []
    adjudication_rows: list[dict[str, Any]] = []
    parsed_responses: list[dict[str, Any]] = []
    response_lookup: dict[str, dict[str, Any]] = {}

    for ledger_row in responses:
        blind_id = blind_response_id(ledger_row)
        output_text = extract_output_text(ledger_row.get("raw_output"))
        parsed: dict[str, Any] | None = None
        parse_status = "not_attempted_incomplete"
        parse_error = ""
        if ledger_row.get("completion_state") == "completed":
            try:
                candidate = json.loads(output_text)
            except json.JSONDecodeError as exc:
                parse_status = "invalid_json"
                parse_error = f"{exc.__class__.__name__}: {exc}"
            else:
                valid, schema_status = validate_structured_output(candidate, ledger_row["material_id"])
                if valid:
                    parsed = candidate
                    parse_status = "parsed_valid"
                else:
                    parse_status = schema_status
        else:
            try:
                json.loads(output_text)
            except json.JSONDecodeError as exc:
                parse_error = f"{exc.__class__.__name__}: {exc}"
            else:
                parse_error = "incomplete response contained parseable JSON but remains excluded by frozen completion rule"

        raw_events: list[dict[str, Any]] = []
        if parsed is not None:
            for index, event in enumerate(parsed.get("events", []), 1):
                copy = dict(event)
                copy["_event_index"] = index
                raw_events.append(copy)
        material_gold = gold_by_material[ledger_row["material_id"]]
        events, duplicates = deduplicate_predicted_events(raw_events)
        events, shared_duplicates = deduplicate_shared_terminal_melts(events, material_gold)
        duplicates.extend(shared_duplicates)
        assignments, ambiguity_records = maximum_cardinality_assignment(events, material_gold)
        assignment_by_pred = {pred_index: (gold_index, label, rationale, operations) for pred_index, gold_index, label, rationale, operations in assignments}
        matched_gold: set[int] = set()
        response_event_rows: list[dict[str, Any]] = []

        for pred_index, event in enumerate(events):
            assigned = assignment_by_pred.get(pred_index)
            gold_index: int | None = None
            unit: dict[str, Any] | None = None
            if assigned:
                gold_index, automatic_label, rationale, operations = assigned
                unit = material_gold[gold_index]
                matched_gold.add(gold_index)
            else:
                automatic_label = "incorrect_identity"
                rationale = "No compatible gold identity after temperature-blind one-to-one assignment."
                operations = list(normalize_state(event.get("starting_form_state")).operations) + list(normalize_state(event.get("product_form_state")).operations)
            needs_adjudication = automatic_label == "partial_ambiguous" or event.get("event_type") == "other"
            adj_id = ""
            final_label = automatic_label
            adjudication_rationale = ""
            second_review_required = False
            gold_unit_assignment_status = "automatic_assignment" if unit else "not_assigned"
            first_review_adjudicated_label = ""
            first_review_gold_unit_id = ""
            consensus_status = "not_applicable"
            use_gold_unit_in_secondary = bool(unit)
            if needs_adjudication:
                reason = "other_family" if event.get("event_type") == "other" else "partial_or_ambiguous_identity"
                adj_id = adjudication_id(blind_id, int(event["_event_index"]), unit["scoring_unit_id"] if unit else None, reason)
                candidate_units = [candidate for candidate in material_gold if candidate["event_type"] == event.get("event_type")]
                if event.get("event_type") == "other":
                    candidate_units = material_gold
                decision = decisions.get(adj_id)
                if decision:
                    final_label = str(decision["adjudicated_label"])
                    adjudication_rationale = str(decision["rationale"])
                    second_review_required = bool(decision.get("second_review_required", True))
                    gold_unit_assignment_status = str(decision.get("gold_unit_assignment_status", "adjudicated_assignment"))
                    first_review_adjudicated_label = str(decision.get("first_review_adjudicated_label", final_label))
                    first_review_gold_unit_id = str(decision.get("first_review_gold_unit_id", decision.get("adjudicated_gold_unit_id") or ""))
                    consensus_status = str(decision.get("consensus_status", "not_recorded"))
                    use_gold_unit_in_secondary = bool(decision.get("use_gold_unit_in_secondary_identity_stability", True))
                    chosen_present = "adjudicated_gold_unit_id" in decision
                    chosen = decision.get("adjudicated_gold_unit_id")
                    if gold_unit_assignment_status == "unresolved_second_review":
                        unit = None
                        gold_index = None
                        use_gold_unit_in_secondary = False
                    elif chosen:
                        candidate = next((item for item in material_gold if item["scoring_unit_id"] == chosen), None)
                        if candidate is None:
                            raise AnalysisError(f"Adjudication {adj_id} names unavailable gold unit {chosen}")
                        unit = candidate
                        gold_index = material_gold.index(candidate)
                    elif chosen_present and final_label == "incorrect_identity":
                        unit = None
                        gold_index = None
                else:
                    final_label = "partial_ambiguous" if automatic_label == "partial_ambiguous" else "incorrect_identity"
                    adjudication_rationale = "Conservative first-review classification pending explicit blinded adjudication."
                    second_review_required = True
                raw_statement = {key: value for key, value in event.items() if not key.startswith("_") and key not in {"temperature", "confidence"}}
                adjudication_rows.append({
                    "adjudication_id": adj_id,
                    "blinded_response_id": blind_id,
                    "material_id": ledger_row["material_id"],
                    "generation": ledger_row["generation"],
                    "predicted_event_index": event["_event_index"],
                    "raw_statement": json.dumps(raw_statement, ensure_ascii=False, sort_keys=True),
                    "canonical_predicted": json.dumps({"event_type": event.get("event_type"), "starting": normalize_state(event.get("starting_form_state")).normalized, "product": normalize_state(event.get("product_form_state")).normalized}, ensure_ascii=False, sort_keys=True),
                    "canonical_candidates": json.dumps([candidate["scoring_unit_id"] for candidate in candidate_units], ensure_ascii=False),
                    "gold_definition": " || ".join(candidate["gold_definition"] for candidate in candidate_units),
                    "initial_automatic_label": automatic_label,
                    "adjudicated_label": final_label,
                    "adjudicated_gold_unit_id": unit["scoring_unit_id"] if unit else "",
                    "first_review_adjudicated_label": first_review_adjudicated_label,
                    "first_review_gold_unit_id": first_review_gold_unit_id,
                    "gold_unit_assignment_status": gold_unit_assignment_status,
                    "consensus_status": consensus_status,
                    "gold_unit_used_in_secondary_identity_stability": str(use_gold_unit_in_secondary and bool(unit)).lower(),
                    "rationale": adjudication_rationale,
                    "second_review_required": str(second_review_required).lower(),
                })
            row = {
                "row_type": "predicted_event",
                "key": ledger_row["key"],
                "blinded_response_id": blind_id,
                "response_id": ledger_row["response_id"],
                "material_id": ledger_row["material_id"],
                "requested_model": ledger_row["requested_model"],
                "generation": ledger_row["generation"],
                "completion_state": ledger_row["completion_state"],
                "parse_status": parse_status,
                "predicted_event_index": event["_event_index"],
                "deduplicated_source_indices": "|".join(map(str, event.get("_original_indices", [event["_event_index"]]))),
                "predicted_event_type": event.get("event_type"),
                "predicted_starting_form_state": event.get("starting_form_state"),
                "predicted_product_form_state": event.get("product_form_state"),
                "predicted_direction": event.get("direction"),
                "predicted_temperature_kind": event.get("temperature", {}).get("kind"),
                "predicted_temperature_value_C": event.get("temperature", {}).get("value_C"),
                "predicted_temperature_lower_C": event.get("temperature", {}).get("lower_C"),
                "predicted_temperature_upper_C": event.get("temperature", {}).get("upper_C"),
                "predicted_temperature_bound": event.get("temperature", {}).get("bound"),
                "confidence": event.get("confidence"),
                "note": event.get("note"),
                "normalization_operations": "|".join(dict.fromkeys(operations)),
                "automatic_label": automatic_label,
                "final_label": final_label,
                "adjudication_id": adj_id,
                "predicted_identity_normalized": "|".join(map(str, predicted_identity_key(event))),
                "gold_unit_assignment_status": gold_unit_assignment_status,
                "match_rationale": rationale,
                "gold_scoring_unit_id": unit["scoring_unit_id"] if unit else "",
                "gold_anchor_ids": "|".join(unit["anchor_ids"]) if unit else "",
                "gold_event_type": unit["event_type"] if unit else "",
                "gold_event_class": unit["event_class"] if unit else "",
                "gold_starting_form_state": "|".join(unit["starting_form_states"]) if unit else "",
                "gold_product_form_state": "|".join(unit["product_form_states"]) if unit else "",
                "gold_definition": unit["gold_definition"] if unit else "",
                "direction_correct": None,
                "temperature_field_compatible": None,
                "temperature_absolute_error_C": None,
                "temperature_absolute_error_field_agnostic_C": None,
            }
            response_event_rows.append(row)

        for duplicate in duplicates:
            response_event_rows.append({
                "row_type": "deduplicated_predicted_event", "key": ledger_row["key"], "blinded_response_id": blind_id,
                "response_id": ledger_row["response_id"], "material_id": ledger_row["material_id"], "requested_model": ledger_row["requested_model"],
                "generation": ledger_row["generation"], "completion_state": ledger_row["completion_state"], "parse_status": parse_status,
                "predicted_event_index": duplicate["_event_index"], "deduplicated_source_indices": str(duplicate["_event_index"]),
                "predicted_event_type": duplicate.get("event_type"), "predicted_starting_form_state": duplicate.get("starting_form_state"),
                "predicted_product_form_state": duplicate.get("product_form_state"), "predicted_direction": duplicate.get("direction"),
                "predicted_temperature_kind": duplicate.get("temperature", {}).get("kind"), "predicted_temperature_value_C": duplicate.get("temperature", {}).get("value_C"),
                "predicted_temperature_lower_C": duplicate.get("temperature", {}).get("lower_C"), "predicted_temperature_upper_C": duplicate.get("temperature", {}).get("upper_C"),
                "predicted_temperature_bound": duplicate.get("temperature", {}).get("bound"), "confidence": duplicate.get("confidence"), "note": duplicate.get("note"),
                "normalization_operations": duplicate.get("_deduplication_reason", "deduplicated"), "automatic_label": "deduplicated", "final_label": "deduplicated",
                "adjudication_id": "", "predicted_identity_normalized": "|".join(map(str, predicted_identity_key(duplicate))),
                "gold_unit_assignment_status": "not_applicable",
                "match_rationale": f"Predicted event deduplicated against event {duplicate['_duplicate_of']} under {duplicate.get('_deduplication_reason', 'deduplication rule')}",
                "gold_scoring_unit_id": "", "gold_anchor_ids": "", "gold_event_type": "", "gold_event_class": "",
                "gold_starting_form_state": "", "gold_product_form_state": "", "gold_definition": "",
                "direction_correct": None, "temperature_field_compatible": None, "temperature_absolute_error_C": None,
                "temperature_absolute_error_field_agnostic_C": None,
            })

        matched_final_gold = {row["gold_scoring_unit_id"] for row in response_event_rows if row.get("final_label") == "correct_identity" and row.get("gold_scoring_unit_id")}
        assigned_any_gold = {row["gold_scoring_unit_id"] for row in response_event_rows if row.get("gold_scoring_unit_id")}
        for unit in material_gold:
            if unit["scoring_unit_id"] in assigned_any_gold:
                continue
            response_event_rows.append({
                "row_type": "unmatched_gold_event", "key": ledger_row["key"], "blinded_response_id": blind_id,
                "response_id": ledger_row["response_id"], "material_id": ledger_row["material_id"], "requested_model": ledger_row["requested_model"],
                "generation": ledger_row["generation"], "completion_state": ledger_row["completion_state"], "parse_status": parse_status,
                "predicted_event_index": "", "deduplicated_source_indices": "", "predicted_event_type": "", "predicted_starting_form_state": "",
                "predicted_product_form_state": "", "predicted_direction": "", "predicted_temperature_kind": "", "predicted_temperature_value_C": "",
                "predicted_temperature_lower_C": "", "predicted_temperature_upper_C": "", "predicted_temperature_bound": "", "confidence": "", "note": "",
                "normalization_operations": "", "automatic_label": "false_negative", "final_label": "false_negative", "adjudication_id": "",
                "predicted_identity_normalized": "", "gold_unit_assignment_status": "not_applicable",
                "match_rationale": "No predicted identity assigned to this gold scoring unit.", "gold_scoring_unit_id": unit["scoring_unit_id"],
                "gold_anchor_ids": "|".join(unit["anchor_ids"]), "gold_event_type": unit["event_type"], "gold_event_class": unit["event_class"],
                "gold_starting_form_state": "|".join(unit["starting_form_states"]), "gold_product_form_state": "|".join(unit["product_form_states"]),
                "gold_definition": unit["gold_definition"], "direction_correct": None, "temperature_field_compatible": None,
                "temperature_absolute_error_C": None, "temperature_absolute_error_field_agnostic_C": None,
            })
        event_rows.extend(response_event_rows)
        parsed_row = {
            **ledger_row,
            "blinded_response_id": blind_id,
            "output_text": output_text,
            "parse_status": parse_status,
            "parse_error": parse_error,
            "parsed_output": parsed,
            "events_after_deduplication": events,
            "event_rows": response_event_rows,
        }
        parsed_responses.append(parsed_row)
        response_lookup[ledger_row["key"]] = parsed_row
    adjudication_rows.sort(key=lambda row: row["adjudication_id"])
    event_rows.sort(key=lambda row: (row["key"], str(row["row_type"]), str(row["predicted_event_index"]), str(row["gold_scoring_unit_id"])))
    return parsed_responses, event_rows, adjudication_rows, response_lookup


def normalize_direction(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"endothermic", "exothermic", "baseline_shift"}:
        return text
    if text == "mixed" or ("endothermic" in text and "exothermic" in text):
        return "mixed"
    return None


def gold_temperature_field(kind: str) -> str | None:
    aliases = {
        "onset": "onset",
        "approximate_onset": "onset",
        "onset_end_intercept": "onset",
        "onset_interval": "onset",
        "onset_range": "onset",
        "midpoint": "midpoint",
        "half_width": "midpoint",
        "peak": "peak",
        "approximate_peak": "peak",
        "mean_peak": "peak",
        "observed_interval": "range",
        "observed_region": "range",
        "start_bound": "bound",
        "upper_bound": "bound",
        # These source labels identify a numeric point without identifying an
        # onset/midpoint/peak convention.  They are recognized explicitly but
        # remain incompatible with a schema field in the primary analysis.
        "approximate_curve_event": "source_specific_point",
        "as_tabulated": "source_specific_point",
        "pooled_value": "source_specific_point",
        "source_event": "source_specific_point",
        "source_transformation": "source_specific_point",
        # A final crystallization temperature has no schema-level synonym.
        "final": "final",
    }
    return aliases.get(str(kind))


def temperature_bound_direction(spec: dict[str, Any], predicted: bool) -> str | None:
    kind = str(spec.get("kind", "")).strip().lower()
    if not predicted:
        if kind == "start_bound":
            return "lower"
        if kind == "upper_bound":
            return "upper"
    text = str(spec.get("bound") or "").strip().lower()
    if any(token in text for token in ("lower", "above", "at least", "minimum", ">")):
        return "lower"
    if any(token in text for token in ("upper", "below", "at most", "maximum", "<")):
        return "upper"
    lower = finite_or_none(spec.get("lower_C"))
    upper = finite_or_none(spec.get("upper_C"))
    if lower is not None and upper is None:
        return "lower"
    if upper is not None and lower is None:
        return "upper"
    return None


def temperature_interval(spec: dict[str, Any], predicted: bool) -> tuple[float, float] | None:
    kind = str(spec.get("kind", ""))
    value = finite_or_none(spec.get("value_C"))
    lower = finite_or_none(spec.get("lower_C"))
    upper = finite_or_none(spec.get("upper_C"))
    if predicted and kind == "range" and lower is not None and upper is not None:
        return (min(lower, upper), max(lower, upper))
    if not predicted and kind == "observed_interval" and lower is not None and upper is not None:
        return (min(lower, upper), max(lower, upper))
    if value is not None:
        return (value, value)
    if lower is not None and upper is not None:
        return (min(lower, upper), max(lower, upper))
    if lower is not None:
        return (lower, lower)
    if upper is not None:
        return (upper, upper)
    return None


def interval_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    if first[1] < second[0]:
        return second[0] - first[1]
    if second[1] < first[0]:
        return first[0] - second[1]
    return 0.0


def one_sided_gold_bound_error(predicted_interval: tuple[float, float], gold_anchor: dict[str, Any]) -> float | None:
    direction = temperature_bound_direction(gold_anchor, predicted=False)
    gold_interval = temperature_interval(gold_anchor, predicted=False)
    if direction is None or gold_interval is None:
        return None
    bound = gold_interval[0]
    if direction == "lower":
        return float(max(0.0, bound - predicted_interval[1]))
    return float(max(0.0, predicted_interval[0] - bound))


def temperature_error(predicted: dict[str, Any], gold_anchor: dict[str, Any], field_agnostic: bool = False) -> tuple[bool, float | None]:
    predicted_kind = str(predicted.get("kind", ""))
    gold_kind = str(gold_anchor.get("kind", ""))
    predicted_field = predicted_kind if predicted_kind in {"onset", "midpoint", "peak", "range", "bound"} else None
    expected_field = gold_temperature_field(gold_kind)
    compatible = predicted_field is not None and expected_field is not None and predicted_field == expected_field
    if not compatible and not field_agnostic:
        return False, None
    pred_interval = temperature_interval(predicted, predicted=True)
    gold_interval = temperature_interval(gold_anchor, predicted=False)
    if pred_interval is None or gold_interval is None:
        return compatible, None
    if expected_field == "bound":
        bound_error = one_sided_gold_bound_error(pred_interval, gold_anchor)
        return compatible, bound_error
    return compatible, float(interval_distance(pred_interval, gold_interval))


def attach_conditional_scores(event_rows: list[dict[str, Any]], gold_units: Sequence[dict[str, Any]]) -> None:
    units = {row["scoring_unit_id"]: row for row in gold_units}
    for row in event_rows:
        if row.get("row_type") != "predicted_event" or row.get("final_label") != "correct_identity":
            continue
        unit = units.get(str(row.get("gold_scoring_unit_id")))
        if unit is None:
            continue
        pred_direction = normalize_direction(row.get("predicted_direction"))
        gold_directions = {normalize_direction(value) for value in unit["directions"]} - {None}
        row["direction_correct"] = (pred_direction in gold_directions) if gold_directions and pred_direction is not None else None
        predicted_temperature = {
            "kind": row.get("predicted_temperature_kind"),
            "value_C": row.get("predicted_temperature_value_C"),
            "lower_C": row.get("predicted_temperature_lower_C"),
            "upper_C": row.get("predicted_temperature_upper_C"),
            "bound": row.get("predicted_temperature_bound"),
        }
        primary_candidates: list[float] = []
        agnostic_candidates: list[float] = []
        any_compatible = False
        for anchor in unit["temperature_anchors"]:
            compatible, primary_error = temperature_error(predicted_temperature, anchor, field_agnostic=False)
            any_compatible = any_compatible or compatible
            if primary_error is not None:
                primary_candidates.append(primary_error)
            _, agnostic_error = temperature_error(predicted_temperature, anchor, field_agnostic=True)
            if agnostic_error is not None:
                agnostic_candidates.append(agnostic_error)
        row["temperature_field_compatible"] = any_compatible
        row["temperature_absolute_error_C"] = min(primary_candidates) if primary_candidates else None
        row["temperature_absolute_error_field_agnostic_C"] = min(agnostic_candidates) if agnostic_candidates else None


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def harmonic(recall: float | None, precision: float | None, gold_n: int, predicted_n: int) -> float:
    if gold_n > 0 and predicted_n == 0:
        return 0.0
    if recall is None or precision is None or recall + precision == 0:
        return 0.0
    return float(2 * recall * precision / (recall + precision))


def create_response_metrics(
    parsed_responses: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    gold_units: Sequence[dict[str, Any]],
    materials_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    events_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    gold_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        events_by_key[row["key"]].append(row)
    for unit in gold_units:
        gold_by_material[unit["material_id"]].append(unit)
    rows: list[dict[str, Any]] = []
    for response in parsed_responses:
        material = materials_by_id[response["material_id"]]
        response_events = [row for row in events_by_key[response["key"]] if row["row_type"] == "predicted_event"]
        gold = gold_by_material[response["material_id"]]
        correct = [row for row in response_events if row["final_label"] == "correct_identity"]
        tp_ids = {row["gold_scoring_unit_id"] for row in correct}
        predicted_n = len(response_events)
        gold_n = len(gold)
        tp = len(tp_ids)
        recall = safe_ratio(tp, gold_n)
        precision = safe_ratio(tp, predicted_n)
        direction_rows = [row for row in correct if row["direction_correct"] is not None]
        temp_errors = [float(row["temperature_absolute_error_C"]) for row in correct if row["temperature_absolute_error_C"] not in (None, "")]
        agnostic_errors = [float(row["temperature_absolute_error_field_agnostic_C"]) for row in correct if row["temperature_absolute_error_field_agnostic_C"] not in (None, "")]
        by_class: dict[str, dict[str, int | float | None]] = {}
        for event_class in ("terminal_melting", "non_melting_pathway"):
            gold_class = [unit for unit in gold if unit["event_class"] == event_class]
            correct_class = [row for row in correct if row["gold_event_class"] == event_class]
            direction_class = [row for row in correct_class if row["direction_correct"] is not None]
            if event_class == "terminal_melting":
                predicted_class = [row for row in response_events if row["predicted_event_type"] == "melting"]
            else:
                predicted_class = [row for row in response_events if row["predicted_event_type"] not in {"melting", "other"}]
            class_recall = safe_ratio(len(correct_class), len(gold_class))
            class_precision = safe_ratio(len(correct_class), len(predicted_class))
            by_class[event_class] = {
                "gold": len(gold_class), "predicted": len(predicted_class), "true_positive": len(correct_class),
                "recall": class_recall, "precision": class_precision,
                "f1": harmonic(class_recall, class_precision, len(gold_class), len(predicted_class)),
                "direction_scorable": len(direction_class),
                "direction_correct": sum(bool(row["direction_correct"]) for row in direction_class),
            }
        usage = response.get("usage", {})
        reasoning = int(usage.get("output_tokens_details", {}).get("reasoning_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        row = {
            "key": response["key"], "blinded_response_id": response["blinded_response_id"], "response_id": response["response_id"],
            "material_id": response["material_id"], "stratum": material["stratum"], "complexity_tier": material["complexity_tier"],
            "requested_model": response["requested_model"], "returned_model": response["returned_model"], "generation": response["generation"],
            "completion_state": response["completion_state"], "incomplete_reason": json.dumps(response.get("incomplete_reason"), sort_keys=True) if response.get("incomplete_reason") else "",
            "parse_status": response["parse_status"], "parse_error": response["parse_error"], "empty_response": predicted_n == 0,
            "gold_events": gold_n, "predicted_events": predicted_n, "correct_identities": tp,
            "partial_ambiguous_events": sum(row["final_label"] == "partial_ambiguous" for row in response_events),
            "incorrect_events": sum(row["final_label"] == "incorrect_identity" for row in response_events),
            "deduplicated_events": sum(row["row_type"] == "deduplicated_predicted_event" for row in events_by_key[response["key"]]),
            "recall": recall, "precision": precision, "f1": harmonic(recall, precision, gold_n, predicted_n),
            "direction_scorable": len(direction_rows), "direction_correct": sum(bool(row["direction_correct"]) for row in direction_rows),
            "direction_accuracy": safe_ratio(sum(bool(row["direction_correct"]) for row in direction_rows), len(direction_rows)),
            "temperature_scorable": len(temp_errors), "temperature_median_absolute_error_C": quantile(temp_errors, 0.5),
            "temperature_mean_absolute_error_C": float(np.mean(temp_errors)) if temp_errors else None,
            "temperature_field_agnostic_scorable": len(agnostic_errors),
            "temperature_field_agnostic_median_absolute_error_C": quantile(agnostic_errors, 0.5),
            "terminal_gold": by_class["terminal_melting"]["gold"], "terminal_predicted": by_class["terminal_melting"]["predicted"],
            "terminal_correct": by_class["terminal_melting"]["true_positive"], "terminal_recall": by_class["terminal_melting"]["recall"],
            "terminal_precision": by_class["terminal_melting"]["precision"], "terminal_f1": by_class["terminal_melting"]["f1"],
            "terminal_direction_scorable": by_class["terminal_melting"]["direction_scorable"],
            "terminal_direction_correct": by_class["terminal_melting"]["direction_correct"],
            "nonmelting_gold": by_class["non_melting_pathway"]["gold"], "nonmelting_predicted": by_class["non_melting_pathway"]["predicted"],
            "nonmelting_correct": by_class["non_melting_pathway"]["true_positive"], "nonmelting_recall": by_class["non_melting_pathway"]["recall"],
            "nonmelting_precision": by_class["non_melting_pathway"]["precision"], "nonmelting_f1": by_class["non_melting_pathway"]["f1"],
            "nonmelting_direction_scorable": by_class["non_melting_pathway"]["direction_scorable"],
            "nonmelting_direction_correct": by_class["non_melting_pathway"]["direction_correct"],
            "unclassified_other_events": sum(row["predicted_event_type"] == "other" for row in response_events),
            "input_tokens": int(usage.get("input_tokens", 0) or 0), "reasoning_tokens": reasoning,
            "nonreasoning_output_tokens": output_tokens - reasoning, "total_output_tokens": output_tokens,
            "total_tokens": int(usage.get("total_tokens", 0) or 0), "visible_json_characters": len(response["output_text"]),
        }
        rows.append(row)
    return rows


def make_gold_outcomes(response_metrics: Sequence[dict[str, Any]], event_rows: Sequence[dict[str, Any]], gold_units: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    units_by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct_by_key: dict[str, set[str]] = defaultdict(set)
    for unit in gold_units:
        units_by_material[unit["material_id"]].append(unit)
    for row in event_rows:
        if row["row_type"] == "predicted_event" and row["final_label"] == "correct_identity" and row["gold_scoring_unit_id"]:
            correct_by_key[row["key"]].add(row["gold_scoring_unit_id"])
    outcomes: list[dict[str, Any]] = []
    for response in response_metrics:
        for unit in units_by_material[response["material_id"]]:
            outcomes.append({
                "key": response["key"], "material_id": response["material_id"], "stratum": response["stratum"],
                "complexity_tier": response["complexity_tier"], "requested_model": response["requested_model"],
                "generation": response["generation"], "completion_state": response["completion_state"], "parse_status": response["parse_status"],
                "scoring_unit_id": unit["scoring_unit_id"], "event_type": unit["event_type"], "event_class": unit["event_class"],
                "protocol_id": unit["protocol_id"], "correct_identity_recovered": int(unit["scoring_unit_id"] in correct_by_key[response["key"]]),
                "counts_toward_primary": unit["counts_toward_primary"],
            })
    return outcomes


def calibration_analysis(
    event_rows: Sequence[dict[str, Any]], response_metrics: Sequence[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stratum_by_key = {row["key"]: row["stratum"] for row in response_metrics}
    summaries: dict[str, dict[str, Any]] = {}
    bin_rows: list[dict[str, Any]] = []
    bins = np.linspace(0.0, 1.0, 11)
    for stratum in STRATA:
        emitted = [
            row for row in event_rows
            if row["row_type"] == "predicted_event"
            and row.get("confidence") not in (None, "")
            and stratum_by_key.get(row["key"]) == stratum
        ]
        confidence = np.asarray([float(row["confidence"]) for row in emitted], dtype=float)
        outcome = np.asarray([1.0 if row["final_label"] == "correct_identity" else 0.0 for row in emitted], dtype=float)
        ece = 0.0
        for index in range(10):
            if index == 9:
                mask = (confidence >= bins[index]) & (confidence <= bins[index + 1])
            else:
                mask = (confidence >= bins[index]) & (confidence < bins[index + 1])
            n = int(mask.sum())
            mean_confidence = float(confidence[mask].mean()) if n else None
            observed = float(outcome[mask].mean()) if n else None
            if n:
                ece += (n / len(emitted)) * abs(float(mean_confidence) - float(observed))
            bin_rows.append({
                "stratum": stratum, "bin_lower": float(bins[index]), "bin_upper": float(bins[index + 1]),
                "n": n, "mean_confidence": mean_confidence, "observed_correct_fraction": observed,
            })
        intercept = slope = intercept_se = slope_se = None
        calibration_status = "estimable"
        if len(np.unique(outcome)) < 2:
            calibration_status = "not_estimable: emitted-event outcomes contain only one class"
        elif len(np.unique(confidence)) < 2:
            calibration_status = "not_estimable: emitted confidences contain only one value"
        else:
            try:
                clipped = np.clip(confidence, 1e-6, 1 - 1e-6)
                logit_confidence = np.log(clipped / (1 - clipped))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.GLM(outcome, sm.add_constant(logit_confidence), family=sm.families.Binomial()).fit()
                intercept, slope = map(float, fit.params)
                intercept_se, slope_se = map(float, fit.bse)
            except Exception as exc:  # sparse or separated calibration fits may be non-estimable
                calibration_status = f"not_estimable: {type(exc).__name__}: {exc}"
        summaries[stratum] = {
            "stratum": stratum,
            "conditioning": "emitted predicted events only; missing gold events are represented through recall",
            "n_emitted": len(emitted), "n_correct": int(outcome.sum()),
            "brier_score": float(np.mean((confidence - outcome) ** 2)) if len(emitted) else None,
            "expected_calibration_error_10_equal_width_bins": float(ece) if len(emitted) else None,
            "calibration_intercept": intercept, "calibration_intercept_se": intercept_se,
            "calibration_slope": slope, "calibration_slope_se": slope_se, "status": calibration_status,
        }
    return summaries, bin_rows


def stability_analysis(
    response_metrics: Sequence[dict[str, Any]], event_rows: Sequence[dict[str, Any]], gold_units: Sequence[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    identities_by_key: dict[str, set[str]] = defaultdict(set)
    temperatures_by_cell_identity: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in event_rows:
        if row["row_type"] != "predicted_event":
            continue
        identity = predicted_identity_from_event_row(row)
        identities_by_key[row["key"]].add(identity)
        if row["final_label"] == "correct_identity" and row.get("temperature_absolute_error_field_agnostic_C") not in (None, ""):
            value = finite_or_none(row.get("predicted_temperature_value_C"))
            if value is None:
                lower = finite_or_none(row.get("predicted_temperature_lower_C"))
                upper = finite_or_none(row.get("predicted_temperature_upper_C"))
                value = (lower + upper) / 2 if lower is not None and upper is not None else None
            if value is not None:
                temperatures_by_cell_identity[(row["material_id"], row["requested_model"], row["gold_scoring_unit_id"])].append(value)
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for metric in response_metrics:
        cells[(metric["material_id"], metric["requested_model"])].append(metric)
    gold_by_material: dict[str, list[str]] = defaultdict(list)
    stratum_by_material: dict[str, str] = {}
    for unit in gold_units:
        gold_by_material[unit["material_id"]].append(unit["scoring_unit_id"])
        stratum_by_material[unit["material_id"]] = unit["stratum"]
    correct_by_key = defaultdict(set)
    for row in event_rows:
        if row["row_type"] == "predicted_event" and row["final_label"] == "correct_identity":
            correct_by_key[row["key"]].add(row["gold_scoring_unit_id"])

    cell_rows: list[dict[str, Any]] = []
    unanimity_rows: list[dict[str, Any]] = []
    for (material, model), members in sorted(cells.items()):
        members = sorted(members, key=lambda row: row["generation"])
        pair_jaccard: list[float] = []
        pair_f1: list[float] = []
        for left, right in combinations(members, 2):
            first = identities_by_key[left["key"]]
            second = identities_by_key[right["key"]]
            union = first | second
            intersection = first & second
            pair_jaccard.append(1.0 if not union else len(intersection) / len(union))
            pair_f1.append(1.0 if not first and not second else 2 * len(intersection) / (len(first) + len(second)))
        for scoring_unit_id in gold_by_material[material]:
            values = [int(scoring_unit_id in correct_by_key[member["key"]]) for member in members]
            unanimity_rows.append({
                "detail_type": "gold_identity_unanimity", "stratum": stratum_by_material[material],
                "material_id": material, "requested_model": model, "scoring_unit_id": scoring_unit_id,
                "recoveries_out_of_3": sum(values), "unanimously_recovered": int(sum(values) == 3),
                "unanimously_missed": int(sum(values) == 0),
            })
        status_pairs = [(member["completion_state"], member["parse_status"]) for member in members]
        cell_rows.append({
            "stratum": members[0]["stratum"], "material_id": material, "requested_model": model, "n_generations": len(members),
            "mean_pairwise_jaccard": float(np.mean(pair_jaccard)), "mean_pairwise_f1": float(np.mean(pair_f1)),
            "min_pairwise_jaccard": float(np.min(pair_jaccard)), "min_pairwise_f1": float(np.min(pair_f1)),
            "parse_failure_status_concordant": int(len(set(status_pairs)) == 1),
            "completion_states": "|".join(member["completion_state"] for member in members),
            "parse_statuses": "|".join(member["parse_status"] for member in members),
        })
    temperature_rows: list[dict[str, Any]] = []
    for (material, model, scoring_unit_id), values in sorted(temperatures_by_cell_identity.items()):
        if len(values) >= 2:
            temperature_rows.append({
                "detail_type": "temperature_stability", "stratum": stratum_by_material[material],
                "material_id": material, "requested_model": model, "scoring_unit_id": scoring_unit_id,
                "n_generations_present": len(values), "temperature_sd_C": float(np.std(values, ddof=1)),
                "temperature_range_C": float(max(values) - min(values)),
            })
    summaries: dict[str, dict[str, Any]] = {}
    for stratum in STRATA:
        stratum_cells = [row for row in cell_rows if row["stratum"] == stratum]
        stratum_unanimity = [row for row in unanimity_rows if row["stratum"] == stratum]
        stratum_temperatures = [row for row in temperature_rows if row["stratum"] == stratum]
        summaries[stratum] = {
            "stratum": stratum,
            "identity_source": "normalized model-produced event statement only; gold assignments never replace predicted identities",
            "cell_count": len(stratum_cells),
            "material_first_summary": {
                "mean_pairwise_jaccard": describe([row["mean_pairwise_jaccard"] for row in stratum_cells]),
                "mean_pairwise_f1": describe([row["mean_pairwise_f1"] for row in stratum_cells]),
                "parse_failure_status_concordance_rate": float(np.mean([row["parse_failure_status_concordant"] for row in stratum_cells])) if stratum_cells else None,
            },
            "gold_identity_unanimity_rate": safe_ratio(sum(row["unanimously_recovered"] for row in stratum_unanimity), len(stratum_unanimity)),
            "gold_identity_unanimous_miss_rate": safe_ratio(sum(row["unanimously_missed"] for row in stratum_unanimity), len(stratum_unanimity)),
            "temperature_stability_identities_n": len(stratum_temperatures),
            "temperature_sd_C": describe([row["temperature_sd_C"] for row in stratum_temperatures]),
            "temperature_range_C": describe([row["temperature_range_C"] for row in stratum_temperatures]),
        }
    return summaries, cell_rows, unanimity_rows + temperature_rows


ADJUDICATION_FIELDS = [
    "adjudication_id", "blinded_response_id", "material_id", "generation", "predicted_event_index",
    "raw_statement", "canonical_predicted", "canonical_candidates", "gold_definition",
    "initial_automatic_label", "adjudicated_label", "adjudicated_gold_unit_id",
    "first_review_adjudicated_label", "first_review_gold_unit_id", "gold_unit_assignment_status",
    "consensus_status", "gold_unit_used_in_secondary_identity_stability", "rationale",
    "second_review_required",
]

RECONCILIATION_FIELDS = [
    "adjudication_id", "material_id", "blinded_response_id", "generation", "predicted_event_index",
    "initial_automatic_label", "first_review_adjudicated_label", "first_review_gold_unit_id",
    "reconciled_label", "reconciled_gold_unit_id", "gold_unit_assignment_status", "consensus_status",
    "gold_unit_used_in_secondary_identity_stability", "reconciliation_rationale",
]

EVENT_MATCH_FIELDS = [
    "row_type", "key", "blinded_response_id", "response_id", "material_id", "requested_model", "generation",
    "completion_state", "parse_status", "predicted_event_index", "deduplicated_source_indices",
    "predicted_event_type", "predicted_starting_form_state", "predicted_product_form_state", "predicted_direction",
    "predicted_temperature_kind", "predicted_temperature_value_C", "predicted_temperature_lower_C",
    "predicted_temperature_upper_C", "predicted_temperature_bound", "confidence", "note",
    "normalization_operations", "automatic_label", "final_label", "adjudication_id",
    "predicted_identity_normalized", "gold_unit_assignment_status", "match_rationale",
    "gold_scoring_unit_id", "gold_anchor_ids", "gold_event_type", "gold_event_class",
    "gold_starting_form_state", "gold_product_form_state", "gold_definition", "direction_correct",
    "temperature_field_compatible", "temperature_absolute_error_C", "temperature_absolute_error_field_agnostic_C",
]


def adjudication_reconciliation_rows(adjudication_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in adjudication_rows:
        if row.get("gold_unit_assignment_status") != "unresolved_second_review":
            continue
        rows.append({
            "adjudication_id": row["adjudication_id"],
            "material_id": row["material_id"],
            "blinded_response_id": row["blinded_response_id"],
            "generation": row["generation"],
            "predicted_event_index": row["predicted_event_index"],
            "initial_automatic_label": row["initial_automatic_label"],
            "first_review_adjudicated_label": row["first_review_adjudicated_label"],
            "first_review_gold_unit_id": row["first_review_gold_unit_id"],
            "reconciled_label": row["adjudicated_label"],
            "reconciled_gold_unit_id": row["adjudicated_gold_unit_id"],
            "gold_unit_assignment_status": row["gold_unit_assignment_status"],
            "consensus_status": row["consensus_status"],
            "gold_unit_used_in_secondary_identity_stability": row["gold_unit_used_in_secondary_identity_stability"],
            "reconciliation_rationale": row["rationale"],
        })
    return rows


def load_analysis_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    manifest, plan, attempts, responses, input_hashes = validate_collection(root)
    materials = load_json(root / "cohort" / "materials.json")
    gold = load_json(root / "cohort" / "literature_reference.json")
    materials_by_id = {row["material_id"]: row for row in materials["materials"]}
    gold_units = make_gold_units(gold, materials_by_id)
    if len(gold_units) != 109:
        raise AnalysisError(f"Expected 109 unique gold scoring units, found {len(gold_units)}")
    if sum(unit["counts_toward_primary"] for unit in gold_units) != 89:
        raise AnalysisError("Expected 89 primary molecular gold scoring units")
    return manifest, materials, gold, plan, attempts, responses, input_hashes


def load_decisions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    value = load_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
        raise AnalysisError(f"Invalid adjudication decision file: {path}")
    decisions: dict[str, dict[str, Any]] = {}
    allowed = {"correct_identity", "partial_ambiguous", "incorrect_identity"}
    for row in value["decisions"]:
        identifier = row.get("adjudication_id")
        if not identifier or identifier in decisions:
            raise AnalysisError("Missing or duplicate adjudication_id in decisions")
        if row.get("adjudicated_label") not in allowed or not str(row.get("rationale", "")).strip():
            raise AnalysisError(f"Incomplete adjudication decision {identifier}")
        decisions[identifier] = row
    return decisions


def prepare_adjudication(root: Path, output_dir: Path) -> dict[str, Any]:
    _, materials, gold, _, _, responses, input_hashes = load_analysis_inputs(root)
    materials_by_id = {row["material_id"]: row for row in materials["materials"]}
    gold_units = make_gold_units(gold, materials_by_id)
    parsed, event_rows, adjudication_rows, _ = parse_and_match(responses, gold_units, decisions={})
    write_csv(output_dir / "blinded_adjudication.csv", adjudication_rows, ADJUDICATION_FIELDS)
    parse_counts = Counter(row["parse_status"] for row in parsed)
    summary = {
        "study_id": STUDY_ID,
        "mode": "prepare_blinded_adjudication",
        "generated_at_utc": ANALYSIS_TIMESTAMP_UTC,
        "responses": len(parsed),
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "adjudication_rows": len(adjudication_rows),
        "automatic_label_counts": dict(sorted(Counter(row["initial_automatic_label"] for row in adjudication_rows).items())),
        "model_arm_fields_excluded": True,
        "temperature_fields_excluded_from_review": True,
        "input_sha256": input_hashes,
    }
    write_json(output_dir / "adjudication_preparation_summary.json", summary)
    after = {
        "manifest": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "MANIFEST.json"),
        "plan": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "execution-plan.jsonl"),
        "attempts": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "attempt-ledger.jsonl"),
        "responses": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "response-ledger.jsonl"),
        "gold": sha256_file(root / "cohort" / "literature_reference.json"),
        "materials": sha256_file(root / "cohort" / "materials.json"),
        "packet": sha256_file(root / "blind" / "MODEL_PACKET.json"),
    }
    if after != input_hashes:
        raise AnalysisError("Raw or frozen input hash changed during adjudication preparation")
    return summary


def bootstrap_material_weights(materials: pd.DataFrame, reps: int, seed: int) -> tuple[list[str], np.ndarray]:
    unique = materials[["material_id", "complexity_tier"]].drop_duplicates().sort_values("material_id")
    material_ids = unique["material_id"].tolist()
    index = {material: i for i, material in enumerate(material_ids)}
    weights = np.zeros((reps, len(material_ids)), dtype=np.int16)
    rng = np.random.default_rng(seed)
    for tier in ("M1", "M2", "M3"):
        members = unique.loc[unique["complexity_tier"] == tier, "material_id"].tolist()
        positions = np.asarray([index[material] for material in members], dtype=int)
        draws = rng.integers(0, len(members), size=(reps, len(members)))
        for replicate in range(reps):
            counts = np.bincount(draws[replicate], minlength=len(members))
            weights[replicate, positions] = counts
    return material_ids, weights


def aggregate_by_material(frame: pd.DataFrame, material_ids: Sequence[str], column: str) -> np.ndarray:
    grouped = frame.groupby("material_id", observed=True)[column].sum()
    return np.asarray([float(grouped.get(material, 0.0)) for material in material_ids])


def ratio_bootstrap(
    frame: pd.DataFrame,
    material_ids: Sequence[str],
    weights: np.ndarray,
    numerator: str,
    denominator: str,
) -> tuple[float | None, np.ndarray]:
    numer = aggregate_by_material(frame, material_ids, numerator)
    denom = aggregate_by_material(frame, material_ids, denominator)
    point = safe_ratio(float(numer.sum()), float(denom.sum()))
    boot_numer = weights @ numer
    boot_denom = weights @ denom
    distribution = np.divide(boot_numer, boot_denom, out=np.full(weights.shape[0], np.nan), where=boot_denom != 0)
    return point, distribution


def interval_from_distribution(distribution: np.ndarray) -> tuple[float | None, float | None]:
    clean = distribution[np.isfinite(distribution)]
    if clean.size == 0:
        return None, None
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def summary_metrics_with_bootstrap(response_metrics: Sequence[dict[str, Any]], reps: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[str], np.ndarray]:
    frame = pd.DataFrame(response_metrics)
    molecular = frame[frame["stratum"] == "molecular_polymorph"].copy()
    material_ids, weights = bootstrap_material_weights(molecular, reps, seed)
    specs: list[tuple[str, str, str, pd.Series]] = []
    true_mask = pd.Series(True, index=molecular.index)
    specs.append(("all", "all", "all", true_mask))
    for model in EXPECTED_MODELS:
        specs.append((model, "all", "all", molecular["requested_model"] == model))
    for tier in ("M1", "M2", "M3"):
        specs.append(("all", tier, "all", molecular["complexity_tier"] == tier))
        for model in EXPECTED_MODELS:
            specs.append((model, tier, "all", (molecular["complexity_tier"] == tier) & (molecular["requested_model"] == model)))
    for event_class in ("terminal_melting", "non_melting_pathway"):
        specs.append(("all", "all", event_class, true_mask))
        for model in EXPECTED_MODELS:
            specs.append((model, "all", event_class, molecular["requested_model"] == model))

    rows: list[dict[str, Any]] = []
    distributions: dict[str, np.ndarray] = {}
    for model, tier, event_class, mask in specs:
        subset = molecular.loc[mask].copy()
        if event_class == "terminal_melting":
            columns = {"gold": "terminal_gold", "predicted": "terminal_predicted", "tp": "terminal_correct"}
            direction_columns = {"correct": "terminal_direction_correct", "scorable": "terminal_direction_scorable"}
        elif event_class == "non_melting_pathway":
            columns = {"gold": "nonmelting_gold", "predicted": "nonmelting_predicted", "tp": "nonmelting_correct"}
            direction_columns = {"correct": "nonmelting_direction_correct", "scorable": "nonmelting_direction_scorable"}
        else:
            columns = {"gold": "gold_events", "predicted": "predicted_events", "tp": "correct_identities"}
            direction_columns = {"correct": "direction_correct", "scorable": "direction_scorable"}
        recall, recall_dist = ratio_bootstrap(subset, material_ids, weights, columns["tp"], columns["gold"])
        precision, precision_dist = ratio_bootstrap(subset, material_ids, weights, columns["tp"], columns["predicted"])
        f1_dist = np.divide(2 * recall_dist * precision_dist, recall_dist + precision_dist, out=np.zeros_like(recall_dist), where=np.isfinite(recall_dist + precision_dist) & ((recall_dist + precision_dist) != 0))
        f1 = harmonic(recall, precision, int(subset[columns["gold"]].sum()), int(subset[columns["predicted"]].sum()))
        direction, direction_dist = ratio_bootstrap(subset, material_ids, weights, direction_columns["correct"], direction_columns["scorable"])
        for metric, point, distribution in (
            ("recall", recall, recall_dist), ("precision", precision, precision_dist), ("f1", f1, f1_dist),
            ("direction_accuracy", direction, direction_dist),
        ):
            low, high = interval_from_distribution(distribution)
            key = f"{model}|{tier}|{event_class}|{metric}"
            distributions[key] = distribution
            if metric == "recall":
                metric_numerator = int(subset[columns["tp"]].sum())
                metric_denominator = int(subset[columns["gold"]].sum())
            elif metric == "precision":
                metric_numerator = int(subset[columns["tp"]].sum())
                metric_denominator = int(subset[columns["predicted"]].sum())
            elif metric == "direction_accuracy":
                metric_numerator = int(subset[direction_columns["correct"]].sum())
                metric_denominator = int(subset[direction_columns["scorable"]].sum())
            else:
                metric_numerator = None
                metric_denominator = None
            rows.append({
                "population": "molecular_polymorph", "requested_model": model, "complexity_tier": tier,
                "event_class": event_class, "metric": metric, "estimate": point,
                "ci95_percentile_low": low, "ci95_percentile_high": high,
                "cluster_bootstrap_reps": reps, "cluster_unit": "material",
                "gold_denominator": int(subset[columns["gold"]].sum()), "predicted_denominator": int(subset[columns["predicted"]].sum()),
                "true_positive_numerator": int(subset[columns["tp"]].sum()),
                "metric_numerator": metric_numerator, "metric_denominator": metric_denominator,
            })
    return rows, distributions, material_ids, weights


DESIGN_NAMES = ["Intercept", "M2", "M3", "gpt-6-luna", "generation_2", "generation_3", "terminal_melting", "gpt-6-luna:M2", "gpt-6-luna:M3"]


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    tier = frame["complexity_tier"].astype(str)
    model6 = (frame["requested_model"] == "gpt-6-luna").astype(float).to_numpy()
    m2 = (tier == "M2").astype(float).to_numpy()
    m3 = (tier == "M3").astype(float).to_numpy()
    generation = frame["generation"].astype(int)
    terminal = (frame["event_class"] == "terminal_melting").astype(float).to_numpy()
    return np.column_stack([
        np.ones(len(frame)), m2, m3, model6,
        (generation == 2).astype(float), (generation == 3).astype(float), terminal,
        model6 * m2, model6 * m3,
    ])


def fit_gee(frame: pd.DataFrame) -> dict[str, Any]:
    X = design_matrix(frame)
    y = frame["correct_identity_recovered"].astype(float).to_numpy()
    groups = frame["material_id"].astype(str).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable()).fit(maxiter=200, cov_type="robust")
    covariance = np.asarray(fit.cov_params(), dtype=float)
    diagonal = np.diag(covariance)
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    inference_valid = bool(np.all(np.isfinite(covariance)) and np.all(diagonal > 0) and np.min(eigenvalues) >= -1e-8)
    standard_errors = np.sqrt(np.where(diagonal >= 0, diagonal, np.nan))
    return {
        "method": "logistic GEE with exchangeable material-cluster working correlation",
        "converged": bool(getattr(fit, "converged", True)),
        "params": np.asarray(fit.params, dtype=float), "cov": covariance,
        "standard_errors": standard_errors, "working_correlation": finite_or_none(getattr(fit.cov_struct, "dep_params", None)),
        "robust_covariance_valid": inference_valid,
        "robust_covariance_minimum_eigenvalue": float(np.min(eigenvalues)),
        "separation_flag": bool(np.max(np.abs(np.asarray(fit.params, dtype=float))) > 15),
        "n_rows": len(frame), "n_material_clusters": frame["material_id"].nunique(),
    }


def fit_glmm(frame: pd.DataFrame, initial_beta: np.ndarray | None = None) -> dict[str, Any]:
    X = design_matrix(frame)
    y = frame["correct_identity_recovered"].astype(float).to_numpy()
    clusters = [np.where(frame["material_id"].astype(str).to_numpy() == material)[0] for material in sorted(frame["material_id"].unique())]
    nodes, quadrature_weights = hermgauss(20)
    log_weights = np.log(quadrature_weights) - 0.5 * math.log(math.pi)

    def objective(theta: np.ndarray) -> float:
        beta = theta[:-1]
        sigma = math.exp(float(theta[-1]))
        total = 0.0
        for indices in clusters:
            eta = X[indices] @ beta
            offsets = math.sqrt(2.0) * sigma * nodes
            logits = eta[:, None] + offsets[None, :]
            log_bernoulli = y[indices, None] * (-np.logaddexp(0.0, -logits)) + (1.0 - y[indices, None]) * (-np.logaddexp(0.0, logits))
            total += float(logsumexp(log_weights + log_bernoulli.sum(axis=0)))
        return -total

    beta0 = np.zeros(X.shape[1]) if initial_beta is None or len(initial_beta) != X.shape[1] else np.asarray(initial_beta)
    start = np.concatenate([beta0, np.asarray([-1.0])])
    result = minimize(
        objective, start, method="L-BFGS-B", bounds=[(None, None)] * X.shape[1] + [(-8.0, 3.0)],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )
    covariance = np.asarray(result.hess_inv.todense(), dtype=float) if hasattr(result.hess_inv, "todense") else np.asarray(result.hess_inv, dtype=float)
    sigma = math.exp(float(result.x[-1]))
    covariance_valid = covariance.shape == (X.shape[1] + 1, X.shape[1] + 1) and np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) > 0)
    separation = bool(np.max(np.abs(result.x[:-1])) > 15)
    singular = sigma < 1e-3 or not covariance_valid
    return {
        "method": "logistic-normal random-intercept GLMM by 20-node adaptive-scale Gauss-Hermite marginal likelihood",
        "optimizer": "L-BFGS-B; one frozen start from GEE fixed effects; maxiter=2000; ftol=1e-12; gtol=1e-8; log-sigma bounds [-8,3]",
        "converged": bool(result.success), "optimizer_message": str(result.message), "singular": bool(singular),
        "separation_flag": separation,
        "random_intercept_sd": sigma, "log_likelihood": float(-result.fun),
        "params": np.asarray(result.x[:-1], dtype=float), "cov": covariance[:-1, :-1] if covariance_valid else np.full((X.shape[1], X.shape[1]), np.nan),
        "standard_errors": np.sqrt(np.diag(covariance[:-1, :-1])) if covariance_valid else np.full(X.shape[1], np.nan),
        "n_rows": len(frame), "n_material_clusters": len(clusters),
    }


def wald_test(params: np.ndarray, covariance: np.ndarray, contrast_matrix: np.ndarray) -> dict[str, Any]:
    estimate = contrast_matrix @ params
    middle = contrast_matrix @ covariance @ contrast_matrix.T
    inverse = np.linalg.pinv(middle)
    statistic = float(estimate.T @ inverse @ estimate)
    df = int(np.linalg.matrix_rank(contrast_matrix))
    return {"statistic": statistic, "df": df, "p_value": float(chi2.sf(statistic, df))}


def linear_contrast(params: np.ndarray, covariance: np.ndarray, vector: np.ndarray, label: str) -> dict[str, Any]:
    estimate = float(vector @ params)
    variance = float(vector @ covariance @ vector)
    se = math.sqrt(max(variance, 0.0))
    z = estimate / se if se > 0 else math.nan
    p_value = float(2 * norm.sf(abs(z))) if math.isfinite(z) else None
    return {
        "contrast": label, "log_odds_difference": estimate, "standard_error": se,
        "odds_ratio": math.exp(estimate), "ci95_low": math.exp(estimate - 1.96 * se),
        "ci95_high": math.exp(estimate + 1.96 * se), "z": z if math.isfinite(z) else None, "p_value": p_value,
    }


def bootstrap_contrast(
    label: str, point: float, distribution: np.ndarray
) -> dict[str, Any]:
    clean = distribution[np.isfinite(distribution)]
    low, high = interval_from_distribution(clean)
    if clean.size:
        tail = min(np.sum(clean <= 0), np.sum(clean >= 0))
        p_value = min(1.0, 2.0 * (tail + 1) / (clean.size + 1))
    else:
        p_value = None
    return {"contrast": label, "risk_difference": point, "ci95_percentile_low": low, "ci95_percentile_high": high, "two_sided_bootstrap_p_value": p_value}


def model_inference(gold_outcomes: Sequence[dict[str, Any]], bootstrap_distributions: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    frame = pd.DataFrame(gold_outcomes)
    primary = frame[(frame["stratum"] == "molecular_polymorph") & (frame["counts_toward_primary"] == True)].copy()  # noqa: E712
    gee = fit_gee(primary)
    glmm = fit_glmm(primary, initial_beta=gee["params"])
    use_glmm = glmm["converged"] and not glmm["singular"] and not glmm["separation_flag"] and np.all(np.isfinite(glmm["cov"]))
    gee_inference_valid = gee["converged"] and gee["robust_covariance_valid"] and not gee["separation_flag"]
    model_based_estimable = use_glmm or gee_inference_valid
    selected = glmm if use_glmm else gee if gee_inference_valid else None
    primary_failure_reason = None
    if not use_glmm:
        primary_failure_reason = (
            "The random-intercept GLMM reached the optimizer stopping rule but fixed effects diverged under complete/quasi-complete separation "
            f"(max |beta|={float(np.max(np.abs(glmm['params']))):.3f}); the prespecified exchangeable GEE also showed separation and an indefinite robust covariance "
            f"(minimum eigenvalue={gee['robust_covariance_minimum_eigenvalue']:.3f}). Model-based Wald intervals and p-values are therefore not estimable."
        )

    tier_points = {tier: float(primary.loc[primary["complexity_tier"] == tier, "correct_identity_recovered"].mean()) for tier in ("M1", "M2", "M3")}
    tier_dist = {tier: bootstrap_distributions[f"all|{tier}|all|recall"] for tier in ("M1", "M2", "M3")}
    complexity_contrasts = [
        bootstrap_contrast("M2 vs M1, marginal risk difference", tier_points["M2"] - tier_points["M1"], tier_dist["M2"] - tier_dist["M1"]),
        bootstrap_contrast("M3 vs M1, marginal risk difference", tier_points["M3"] - tier_points["M1"], tier_dist["M3"] - tier_dist["M1"]),
        bootstrap_contrast("M3 vs M2, marginal risk difference", tier_points["M3"] - tier_points["M2"], tier_dist["M3"] - tier_dist["M2"]),
    ]
    adjusted = holm_adjust([float(row["two_sided_bootstrap_p_value"]) for row in complexity_contrasts])
    for row, value in zip(complexity_contrasts, adjusted):
        row["holm_adjusted_bootstrap_p_value"] = value
    model_contrasts = []
    model_difference_distributions: dict[str, np.ndarray] = {}
    for tier in ("M1", "M2", "M3"):
        first = bootstrap_distributions[f"gpt-5.6-luna|{tier}|all|recall"]
        second = bootstrap_distributions[f"gpt-6-luna|{tier}|all|recall"]
        point_first = float(primary.loc[(primary["complexity_tier"] == tier) & (primary["requested_model"] == "gpt-5.6-luna"), "correct_identity_recovered"].mean())
        point_second = float(primary.loc[(primary["complexity_tier"] == tier) & (primary["requested_model"] == "gpt-6-luna"), "correct_identity_recovered"].mean())
        distribution = second - first
        model_difference_distributions[tier] = distribution
        model_contrasts.append(bootstrap_contrast(f"gpt-6-luna vs gpt-5.6-luna within {tier}, marginal risk difference", point_second - point_first, distribution))
    adjusted = holm_adjust([float(row["two_sided_bootstrap_p_value"]) for row in model_contrasts])
    for row, value in zip(model_contrasts, adjusted):
        row["holm_adjusted_bootstrap_p_value"] = value

    if model_based_estimable and selected is not None:
        contrast = np.zeros((2, len(DESIGN_NAMES)))
        contrast[0, DESIGN_NAMES.index("M2")] = 1.0
        contrast[1, DESIGN_NAMES.index("M3")] = 1.0
        joint_values = wald_test(selected["params"], selected["cov"], contrast)
        joint_complexity = {
            "status": "estimable", "method": selected["method"], "estimable": True,
            **joint_values,
        }
    else:
        joint_complexity = {
            "status": "not_estimable", "method": "prespecified random-intercept GLMM with sole exchangeable material-cluster GEE fallback",
            "estimable": False, "statistic": None, "df": 2, "p_value": None,
            "reason": primary_failure_reason,
        }

    interaction_contrasts = [
        bootstrap_contrast(
            "(gpt-6-luna minus gpt-5.6-luna) M2 minus M1, marginal risk-difference contrast",
            float(model_contrasts[1]["risk_difference"] - model_contrasts[0]["risk_difference"]),
            model_difference_distributions["M2"] - model_difference_distributions["M1"],
        ),
        bootstrap_contrast(
            "(gpt-6-luna minus gpt-5.6-luna) M3 minus M1, marginal risk-difference contrast",
            float(model_contrasts[2]["risk_difference"] - model_contrasts[0]["risk_difference"]),
            model_difference_distributions["M3"] - model_difference_distributions["M1"],
        ),
    ]
    interaction_adjusted = holm_adjust([float(row["two_sided_bootstrap_p_value"]) for row in interaction_contrasts])
    for row, value in zip(interaction_contrasts, interaction_adjusted):
        row["holm_adjusted_bootstrap_p_value"] = value

    coefficients = []
    diagnostic_source = glmm if use_glmm else gee
    for name, estimate, se in zip(DESIGN_NAMES, diagnostic_source["params"], diagnostic_source["standard_errors"]):
        valid_se = finite_or_none(se)
        coefficients.append({
            "term": name, "estimate_log_odds": float(estimate), "standard_error": valid_se,
            "model_based_inference_estimable": bool(model_based_estimable and valid_se is not None),
        })
    primary_results = {
        "selected_model": selected["method"] if selected is not None else None,
        "GEE_fallback_attempted": not use_glmm,
        "model_based_inference_estimable": model_based_estimable,
        "fallback_reason": primary_failure_reason,
        "joint_primary_M1_M3_complexity_test": joint_complexity,
        "secondary_model_by_complexity_interaction_test": {
            "status": "not_estimable_under_prespecified_models" if not model_based_estimable else "estimable_under_prespecified_model",
            "model_based_interaction_test_estimable": model_based_estimable,
            "p_value": None,
        },
        "exploratory_unprespecified_marginal_cluster_bootstrap": {
            "prespecified": False,
            "inferential_role": "exploratory only; does not supply the frozen primary p-value",
            "complexity_pairwise_contrasts": complexity_contrasts,
            "model_contrasts_by_complexity": model_contrasts,
            "model_by_complexity_difference_in_difference_contrasts": interaction_contrasts,
        },
        "coefficients": coefficients,
        "mixed_model_attempt": {**{key: value for key, value in glmm.items() if key not in {"params", "cov", "standard_errors"}}, "coefficients": [{"term": name, "estimate_log_odds": float(value), "standard_error": finite_or_none(se)} for name, value, se in zip(DESIGN_NAMES, glmm["params"], glmm["standard_errors"])]},
        "GEE_fallback": {**{key: value for key, value in gee.items() if key not in {"params", "cov", "standard_errors"}}, "coefficients": coefficients},
    }
    gee_serialized = {
        "status": "not_estimable_for_robust_inference" if not gee_inference_valid else "estimable",
        "reason": None if gee_inference_valid else primary_failure_reason,
        "fit_details": {key: value for key, value in gee.items() if key not in {"params", "cov", "standard_errors"}},
        "coefficients": [{"term": name, "estimate_log_odds": float(value), "standard_error": finite_or_none(se)} for name, value, se in zip(DESIGN_NAMES, gee["params"], gee["standard_errors"])],
        "joint_primary_M1_M3_complexity_test": {"estimable": False, "reason": "robust covariance is indefinite under separation"} if not gee_inference_valid else None,
    }
    return primary_results, gee_serialized


def bca_interval(point: float, bootstrap: np.ndarray, jackknife: np.ndarray) -> dict[str, Any]:
    boot = bootstrap[np.isfinite(bootstrap)]
    jack = jackknife[np.isfinite(jackknife)]
    if boot.size < 100 or jack.size < 3:
        return {"stable": False, "reason": "insufficient finite bootstrap or jackknife values", "low": None, "high": None}
    proportion = (np.sum(boot < point) + 0.5 * np.sum(boot == point)) / boot.size
    if proportion <= 0 or proportion >= 1:
        return {"stable": False, "reason": "bias-correction quantile is infinite", "low": None, "high": None}
    z0 = float(norm.ppf(proportion))
    mean_jack = float(np.mean(jack))
    numerator = float(np.sum((mean_jack - jack) ** 3))
    denominator = float(6 * np.sum((mean_jack - jack) ** 2) ** 1.5)
    if denominator == 0 or not math.isfinite(denominator):
        return {"stable": False, "reason": "jackknife acceleration denominator is zero or nonfinite", "low": None, "high": None}
    acceleration = numerator / denominator
    probabilities: list[float] = []
    for alpha in (0.025, 0.975):
        z_alpha = float(norm.ppf(alpha))
        adjusted = float(norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha))))
        probabilities.append(adjusted)
    if not (0 <= probabilities[0] < probabilities[1] <= 1):
        return {"stable": False, "reason": f"invalid adjusted probabilities {probabilities}", "low": None, "high": None}
    return {
        "stable": True, "reason": None, "low": float(np.quantile(boot, probabilities[0])), "high": float(np.quantile(boot, probabilities[1])),
        "bias_correction_z0": z0, "acceleration": acceleration, "adjusted_probabilities": probabilities,
    }


def sensitivity_analyses(
    response_metrics: Sequence[dict[str, Any]],
    event_rows: Sequence[dict[str, Any]],
    gold_units: Sequence[dict[str, Any]],
    gold_outcomes: Sequence[dict[str, Any]],
    protocols: Sequence[dict[str, Any]],
    bootstrap_distributions: dict[str, np.ndarray],
    gee_result: dict[str, Any],
) -> dict[str, Any]:
    response = pd.DataFrame(response_metrics)
    molecular = response[response["stratum"] == "molecular_polymorph"].copy()
    events = pd.DataFrame(event_rows)
    predicted = events[(events["row_type"] == "predicted_event") & (events["material_id"].str.startswith("mol_"))].copy()
    outcomes = pd.DataFrame(gold_outcomes)
    primary_outcomes = outcomes[(outcomes["stratum"] == "molecular_polymorph") & (outcomes["counts_toward_primary"] == True)].copy()  # noqa: E712
    gold_lookup = {unit["scoring_unit_id"]: unit for unit in gold_units}

    half_tp = float((predicted["final_label"] == "correct_identity").sum()) + 0.5 * float((predicted["final_label"] == "partial_ambiguous").sum())
    gold_total = float(molecular["gold_events"].sum())
    predicted_total = float(molecular["predicted_events"].sum())
    half_recall = safe_ratio(half_tp, gold_total)
    half_precision = safe_ratio(half_tp, predicted_total)

    agnostic = pd.to_numeric(predicted.loc[predicted["final_label"] == "correct_identity", "temperature_absolute_error_field_agnostic_C"], errors="coerce").dropna().to_numpy()
    primary_temp = pd.to_numeric(predicted.loc[predicted["final_label"] == "correct_identity", "temperature_absolute_error_C"], errors="coerce").dropna().to_numpy()

    incomplete_protocol_ids: set[str] = set()
    for protocol in protocols:
        if any(value is None or (isinstance(value, str) and "not reported" in value.lower()) for key, value in protocol.items() if key != "protocol_id"):
            incomplete_protocol_ids.add(protocol["protocol_id"])
    complete_units = {unit["scoring_unit_id"] for unit in gold_units if unit["protocol_id"] not in incomplete_protocol_ids}
    complete = primary_outcomes[primary_outcomes["scoring_unit_id"].isin(complete_units)]
    complete_recall = safe_ratio(float(complete["correct_identity_recovered"].sum()), float(len(complete)))

    interval_or_bound_kinds = {
        "observed_interval", "observed_region", "onset_interval", "onset_range", "start_bound", "upper_bound",
    }
    interval_units = {
        unit["scoring_unit_id"] for unit in gold_units
        if any(anchor.get("kind") in interval_or_bound_kinds for anchor in unit["temperature_anchors"])
    }
    point_only = primary_outcomes[~primary_outcomes["scoring_unit_id"].isin(interval_units)]
    point_only_recall = safe_ratio(float(point_only["correct_identity_recovered"].sum()), float(len(point_only)))

    per_material = primary_outcomes.groupby("material_id", observed=True)["correct_identity_recovered"].mean()
    event_weighted = safe_ratio(float(primary_outcomes["correct_identity_recovered"].sum()), float(len(primary_outcomes)))
    equal_material = float(per_material.mean())

    full_recall = float(primary_outcomes["correct_identity_recovered"].mean())
    influence_rows: list[dict[str, Any]] = []
    for material in sorted(primary_outcomes["material_id"].unique()):
        reduced = primary_outcomes[primary_outcomes["material_id"] != material]
        estimate = float(reduced["correct_identity_recovered"].mean())
        influence_rows.append({"material_id": material, "recall_without_material": estimate, "change_from_full": estimate - full_recall})

    bca_rows: list[dict[str, Any]] = []
    for tier in ("M1", "M2", "M3"):
        tier_frame = primary_outcomes[primary_outcomes["complexity_tier"] == tier]
        point = float(tier_frame["correct_identity_recovered"].mean())
        material_values = tier_frame.groupby("material_id", observed=True)["correct_identity_recovered"].agg(["sum", "count"])
        jackknife: list[float] = []
        for material in material_values.index:
            numerator = float(material_values["sum"].sum() - material_values.loc[material, "sum"])
            denominator = float(material_values["count"].sum() - material_values.loc[material, "count"])
            if denominator:
                jackknife.append(numerator / denominator)
        distribution = bootstrap_distributions[f"all|{tier}|all|recall"]
        bca_rows.append({"complexity_tier": tier, "point_estimate": point, **bca_interval(point, distribution, np.asarray(jackknife))})

    mismatch_n = int((response["requested_model"] != response["returned_model"]).sum())
    parseable = molecular[molecular["parse_status"] == "parsed_valid"]
    parseable_recall = safe_ratio(float(parseable["correct_identities"].sum()), float(parseable["gold_events"].sum()))

    return {
        "1_partial_ambiguous_half_credit": {
            "status": "estimable", "half_credit_true_positive_equivalent": half_tp,
            "recall": half_recall, "precision": half_precision,
            "f1": harmonic(half_recall, half_precision, int(gold_total), int(predicted_total)),
        },
        "2_field_agnostic_temperature_error": {
            "status": "estimable", "n": int(len(agnostic)), "median_absolute_error_C": quantile(agnostic, 0.5),
            "iqr_C": [quantile(agnostic, 0.25), quantile(agnostic, 0.75)],
            "primary_field_compatible_n": int(len(primary_temp)), "primary_median_absolute_error_C": quantile(primary_temp, 0.5),
        },
        "3_exclude_protocols_with_not_reported": {
            "status": "estimable", "excluded_protocol_count": len(incomplete_protocol_ids),
            "included_gold_response_rows": int(len(complete)), "recall": complete_recall,
        },
        "4_exclude_source_intervals_or_bounds": {
            "status": "estimable", "excluded_gold_unit_count": len(interval_units),
            "included_gold_response_rows": int(len(point_only)), "recall": point_only_recall,
        },
        "5_equal_weight_materials": {
            "status": "estimable", "equal_material_weight_recall": equal_material,
            "gold_event_weighted_recall": event_weighted, "material_count": int(len(per_material)),
        },
        "6_leave_one_material_out": {
            "status": "estimable", "full_recall": full_recall, "rows": influence_rows,
            "largest_absolute_change": max(influence_rows, key=lambda row: abs(row["change_from_full"])),
        },
        "7_BCa_cluster_bootstrap": {
            "status": "estimable_with_stability_check", "rows": bca_rows,
            "primary_interval_rule": "percentile intervals retained as primary regardless of BCa stability",
        },
        "8_GEE_fallback_comparison": {"status": "estimable", **gee_result},
        "9_exclude_model_identifier_mismatch": {
            "status": "estimable_but_no_records_excluded", "mismatch_response_count": mismatch_n,
            "result_identical_to_primary": mismatch_n == 0,
        },
        "10_parseable_outputs_only": {
            "status": "estimable_selection_prone", "response_count": int(len(parseable)),
            "excluded_response_count": int(len(molecular) - len(parseable)), "recall": parseable_recall,
        },
    }


def token_distribution_rows(response_metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(response_metrics)
    groupings: list[tuple[str, str, pd.Series]] = [("overall", "all", pd.Series(True, index=frame.index))]
    for model in EXPECTED_MODELS:
        groupings.append(("model", model, frame["requested_model"] == model))
    for tier in sorted(frame["complexity_tier"].unique()):
        groupings.append(("complexity", str(tier), frame["complexity_tier"] == tier))
    for state in sorted(frame["completion_state"].unique()):
        groupings.append(("completion_state", str(state), frame["completion_state"] == state))
    measures = ["input_tokens", "reasoning_tokens", "nonreasoning_output_tokens", "total_output_tokens", "total_tokens", "visible_json_characters"]
    rows: list[dict[str, Any]] = []
    for dimension, value, mask in groupings:
        subset = frame.loc[mask]
        for measure in measures:
            stats = describe(subset[measure].astype(float).tolist())
            rows.append({"group_dimension": dimension, "group_value": value, "measure": measure, **stats})
    return rows


def round_up(value: float, quantum: int = 128) -> int:
    return int(math.ceil(float(value) / quantum) * quantum)


def cost_analysis(response_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(response_metrics)
    observed_luna_arms: dict[str, Any] = {}
    observed_luna_total = 0.0
    for model in EXPECTED_MODELS:
        subset = frame[frame["requested_model"] == model]
        input_tokens = int(subset["input_tokens"].sum())
        output_tokens = int(subset["total_output_tokens"].sum())
        rates = MODELS_LUNA[model]
        cost = (input_tokens * rates["input_per_million_usd"] + output_tokens * rates["output_per_million_usd"]) / 1_000_000
        observed_luna_arms[model] = {
            "requests": int(len(subset)), "input_tokens": input_tokens, "total_output_tokens": output_tokens,
            "frozen_rate_usd_per_million": rates, "cost_usd": cost,
        }
        observed_luna_total += cost
    same_token_arms: dict[str, Any] = {}
    total_standard = 0.0
    for sol_model, source_model in (("gpt-5.6-sol", "gpt-5.6-luna"), ("gpt-6-sol", "gpt-6-luna")):
        subset = frame[frame["requested_model"] == source_model]
        input_tokens = int(subset["input_tokens"].sum())
        output_tokens = int(subset["total_output_tokens"].sum())
        rates = MODELS_SOL[sol_model]
        standard = (input_tokens * rates["input_per_million_usd"] + output_tokens * rates["output_per_million_usd"]) / 1_000_000
        same_token_arms[sol_model] = {
            "source_observed_arm": source_model, "requests": int(len(subset)), "observed_input_tokens": input_tokens,
            "observed_total_output_tokens": output_tokens, "standard_same_token_cost_usd": standard,
            "batch_same_token_cost_usd": standard * 0.5, "flex_same_token_cost_usd": standard * 0.5,
            "batch_flex_operational_assumption": "same token counts, eligible workload, and official 50% processing-tier rates; latency/queue behavior not measured",
        }
        total_standard += standard
    completed = frame[frame["completion_state"] == "completed"]
    visible_proxy = describe(completed["nonreasoning_output_tokens"].tolist())
    reasoning_medium = describe(completed["reasoning_tokens"].tolist())
    total_output_completed = describe(completed["total_output_tokens"].tolist())
    visible_caps = {name: round_up(visible_proxy[name]) for name in ("p90", "p95", "p99", "max")}
    reasoning_allowances = {
        "none": 0,
        "low_planning_allowance": round_up(reasoning_medium["median"]),
        "medium_planning_allowance": round_up(reasoning_medium["p95"]),
    }
    scenarios: dict[str, Any] = {}
    for sol_model, source_model in (("gpt-5.6-sol", "gpt-5.6-luna"), ("gpt-6-sol", "gpt-6-luna")):
        subset = frame[frame["requested_model"] == source_model]
        n = len(subset)
        input_tokens = int(subset["input_tokens"].sum())
        rates = MODELS_SOL[sol_model]
        def reservation(output_per_request: int) -> float:
            return (input_tokens * rates["input_per_million_usd"] + n * output_per_request * rates["output_per_million_usd"]) / 1_000_000
        optimistic_output = round_up(visible_proxy["median"])
        conservative_output = round_up(visible_caps["p99"] + reasoning_allowances["low_planning_allowance"], 256)
        medium_output = round_up(visible_caps["p99"] + reasoning_allowances["medium_planning_allowance"], 256)
        scenarios[sol_model] = {
            "optimistic_floor_none_reasoning": {"reasoning_effort": "none", "output_tokens_reserved_per_request": optimistic_output, "cost_usd": reservation(optimistic_output), "assumption": "observed completed-response median nonreasoning output proxy and zero reasoning tokens"},
            "conservative_planning_low": {"reasoning_effort": "low", "output_tokens_reserved_per_request": conservative_output, "cost_usd": reservation(conservative_output), "assumption": "observed p99 nonreasoning output proxy plus observed-medium median reasoning as an explicitly unmeasured low-effort allowance"},
            "medium_planning": {"reasoning_effort": "medium", "output_tokens_reserved_per_request": medium_output, "cost_usd": reservation(medium_output), "assumption": "observed p99 nonreasoning output proxy plus observed-medium p95 reasoning allowance; Sol behavior is unmeasured"},
            "hard_current_4000_reservation": {"reasoning_effort": "unspecified", "output_tokens_reserved_per_request": 4000, "cost_usd": reservation(4000), "assumption": "all requests consume the full frozen 4,000-token generation ceiling"},
        }
    recommended_cap = max(4096, round_up(visible_caps["p99"] + reasoning_allowances["low_planning_allowance"], 256))
    return {
        "observed_facts": {
            "requests": len(frame), "completed": int((frame["completion_state"] == "completed").sum()),
            "incomplete": int((frame["completion_state"] == "incomplete").sum()),
            "luna_cost_by_arm": observed_luna_arms, "luna_cost_usd": observed_luna_total,
            "input_tokens": describe(frame["input_tokens"].tolist()), "reasoning_tokens": describe(frame["reasoning_tokens"].tolist()),
            "usage_derived_nonreasoning_output_tokens": describe(frame["nonreasoning_output_tokens"].tolist()),
            "total_output_tokens": describe(frame["total_output_tokens"].tolist()),
            "visible_json_characters": describe(frame["visible_json_characters"].tolist()),
            "completed_only_nonreasoning_output_proxy": visible_proxy, "completed_only_total_output_tokens": total_output_completed,
            "measurement_note": "output_tokens minus itemized reasoning_tokens is a usage-derived nonreasoning-output proxy, not an exact visible-token count; official documentation states formatting tokens may be unitemized.",
        },
        "official_pricing_and_controls_verified_2026_09_27": {
            "models": MODELS_SOL, "reasoning_efforts_supported_by_both": ["none", "low", "medium", "high", "xhigh", "max"],
            "default_reasoning_effort": "medium", "max_output_tokens_includes": ["reasoning tokens", "visible output tokens", "non-visible formatting tokens"],
            "batch_and_flex_rate_fraction_of_standard": 0.5, "sources": OFFICIAL_DOCS,
        },
        "exact_same_token_counterfactual": {
            "arms": same_token_arms, "two_arm_222_request_standard_cost_usd": total_standard,
            "two_arm_batch_cost_usd": total_standard * 0.5, "two_arm_flex_cost_usd": total_standard * 0.5,
            "interpretation": "Counterfactual price substitution on observed Luna token counts; no Sol token usage was observed.",
        },
        "bounded_scenarios": scenarios,
        "output_ceiling_analysis": {
            "completed_nonreasoning_proxy_caps_tokens": visible_caps, "reasoning_allowances_tokens": reasoning_allowances,
            "recommended_future_low_effort_fail_closed_cap_tokens": recommended_cap,
            "why_headroom_is_required": "Reasoning and non-visible formatting tokens count against max_output_tokens before or alongside the visible JSON.",
        },
        "future_pilot_recommendation": {
            "design": "Before a full matched 222-request Sol comparison, run an authorization-gated 18-request settings pilot: three prespecified molecular materials per M1-M3 tier, one fresh request per Sol arm, with frozen prompts/schema, Standard processing, reasoning.effort='low', and the fail-closed cap above. Evaluate only completion, schema validity, and token use before freezing the full run.",
            "non_goals": "Do not use pilot accuracy to alter the cohort, gold, scientific endpoint, or analysis plan.",
            "manual_gates": ["explicit API-run authorization and cost cap", "independent review of adjudications", "freeze exact Sol identifiers, effort, cap, and execution order"],
        },
    }


def conditional_temperature_summary(event_rows: Sequence[dict[str, Any]], material_ids: Sequence[str], weights: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame(event_rows)
    frame = frame[(frame["row_type"] == "predicted_event") & (frame["final_label"] == "correct_identity") & frame["material_id"].str.startswith("mol_")].copy()
    frame["error"] = pd.to_numeric(frame["temperature_absolute_error_C"], errors="coerce")
    specs: list[tuple[str, str, pd.Series]] = [("all", "all", pd.Series(True, index=frame.index))]
    for event_class in ("terminal_melting", "non_melting_pathway"):
        specs.append((event_class, "all", frame["gold_event_class"] == event_class))
    for field in ("onset", "midpoint", "peak", "range", "bound"):
        specs.append(("all", field, frame["predicted_temperature_kind"] == field))
    rows: list[dict[str, Any]] = []
    for event_class, field, mask in specs:
        subset = frame.loc[mask & frame["error"].notna()].copy()
        material_means = subset.groupby("material_id", observed=True)["error"].mean()
        values = np.asarray([float(material_means.get(material, 0.0)) for material in material_ids])
        present = np.asarray([1.0 if material in material_means.index else 0.0 for material in material_ids])
        denominator = weights @ present
        distribution = np.divide(weights @ values, denominator, out=np.full(weights.shape[0], np.nan), where=denominator != 0)
        low, high = interval_from_distribution(distribution)
        rows.append({
            "population": "molecular_polymorph", "event_class": event_class, "temperature_field": field,
            "identity_and_field_compatible_n": int(len(subset)), "material_n": int(len(material_means)),
            "median_absolute_error_C": quantile(subset["error"].tolist(), 0.5),
            "iqr_low_C": quantile(subset["error"].tolist(), 0.25), "iqr_high_C": quantile(subset["error"].tolist(), 0.75),
            "material_level_mean_absolute_error_C": float(material_means.mean()) if len(material_means) else None,
            "material_level_mean_ci95_percentile_low_C": low, "material_level_mean_ci95_percentile_high_C": high,
            "cluster_bootstrap_reps": int(weights.shape[0]),
        })
    return rows


def descriptive_strata(response_metrics: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    frame = pd.DataFrame(response_metrics)
    rows: list[dict[str, Any]] = []
    for stratum in ("polymer_control", "complex_material"):
        subset = frame[frame["stratum"] == stratum]
        for keys, group in subset.groupby(["material_id", "complexity_tier", "requested_model"], observed=True):
            material, tier, model = keys
            tp = int(group["correct_identities"].sum())
            gold = int(group["gold_events"].sum())
            predicted = int(group["predicted_events"].sum())
            recall = safe_ratio(tp, gold)
            precision = safe_ratio(tp, predicted)
            rows.append({
                "stratum": stratum, "material_id": material, "complexity_tier": tier, "requested_model": model,
                "responses": int(len(group)), "completed": int((group["completion_state"] == "completed").sum()),
                "gold_event_opportunities": gold, "predicted_events": predicted, "correct_identities": tp,
                "recall": recall, "precision": precision, "f1": harmonic(recall, precision, gold, predicted),
            })
    polymer_rows = [row for row in rows if row["stratum"] == "polymer_control"]
    cocoa_rows = [row for row in rows if row["stratum"] == "complex_material"]
    polymer = {
        "scope": "descriptive prospective polymer ladder; excluded from molecular pooled inference",
        "rows": polymer_rows,
        "by_tier": [],
    }
    for tier in ("P1", "P2", "P3"):
        tier_rows = [row for row in polymer_rows if row["complexity_tier"] == tier]
        polymer["by_tier"].append({
            "complexity_tier": tier, "material_count": len({row["material_id"] for row in tier_rows}),
            "gold_event_opportunities": sum(row["gold_event_opportunities"] for row in tier_rows),
            "correct_identities": sum(row["correct_identities"] for row in tier_rows),
            "recall": safe_ratio(sum(row["correct_identities"] for row in tier_rows), sum(row["gold_event_opportunities"] for row in tier_rows)),
        })
    cocoa = {"scope": "single multicomponent polymorphic cocoa-butter case; excluded from pure-compound inference", "rows": cocoa_rows}
    return rows, polymer, cocoa


COLORS = {"gpt-5.6-luna": "#0072B2", "gpt-6-luna": "#D55E00", "terminal_melting": "#009E73", "non_melting_pathway": "#CC79A7"}


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(
        output_dir / f"{stem}.svg", bbox_inches="tight",
        metadata={"Creator": "analysis/analyze_dsc_benchmark.py", "Date": ANALYSIS_TIMESTAMP_UTC},
    )
    fig.savefig(
        output_dir / f"{stem}.png", dpi=220, bbox_inches="tight",
        metadata={"Software": "analysis/analyze_dsc_benchmark.py"},
    )
    plt.close(fig)


def create_figures(
    output_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    stability_cells: Sequence[dict[str, Any]],
    calibration_bins: Sequence[dict[str, Any]],
    cost: dict[str, Any],
) -> list[str]:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8,
        "figure.titlesize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": STUDY_ID,
    })
    summary = pd.DataFrame(summary_rows)
    generated: list[str] = []

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    tiers = ["M1", "M2", "M3"]
    offsets = {"gpt-5.6-luna": -0.10, "gpt-6-luna": 0.10}
    for model in EXPECTED_MODELS:
        rows = summary[(summary["metric"] == "recall") & (summary["event_class"] == "all") & (summary["requested_model"] == model) & summary["complexity_tier"].isin(tiers)].set_index("complexity_tier")
        x = np.arange(3) + offsets[model]
        y = np.asarray([rows.loc[tier, "estimate"] for tier in tiers], dtype=float)
        low = np.asarray([rows.loc[tier, "ci95_percentile_low"] for tier in tiers], dtype=float)
        high = np.asarray([rows.loc[tier, "ci95_percentile_high"] for tier in tiers], dtype=float)
        ax.errorbar(x, y, yerr=np.vstack([y - low, high - y]), fmt="o", ms=6, capsize=3, color=COLORS[model], label=model)
    ax.set_xticks(np.arange(3), tiers)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Identity-aware recall")
    ax.set_xlabel("Prespecified molecular complexity class")
    ax.set_title("Descriptive molecular event-identity recovery by complexity and model")
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.text(0.01, 0.01, "Points are descriptive event recall; bars are 95% percentile material-cluster bootstrap intervals. The frozen primary GLMM/GEE test was not estimable.", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, output_dir, "primary_complexity_model")
    generated.extend(["primary_complexity_model.svg", "primary_complexity_model.png"])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    event_rows = summary[(summary["metric"] == "recall") & (summary["complexity_tier"] == "all") & (summary["requested_model"] == "all") & summary["event_class"].isin(["terminal_melting", "non_melting_pathway"])].set_index("event_class")
    labels = ["Terminal melting", "Non-melting pathway"]
    classes = ["terminal_melting", "non_melting_pathway"]
    y = [event_rows.loc[item, "estimate"] for item in classes]
    low = [event_rows.loc[item, "ci95_percentile_low"] for item in classes]
    high = [event_rows.loc[item, "ci95_percentile_high"] for item in classes]
    ax.bar(np.arange(2), y, color=[COLORS[item] for item in classes], width=0.62)
    ax.errorbar(np.arange(2), y, yerr=np.vstack([np.asarray(y) - np.asarray(low), np.asarray(high) - np.asarray(y)]), fmt="none", ecolor="black", capsize=3)
    ax.set_xticks(np.arange(2), labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Identity-aware recall")
    ax.set_title("Recovery of terminal melts and non-melting pathway events")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output_dir, "event_class_recovery")
    generated.extend(["event_class_recovery.svg", "event_class_recovery.png"])

    stability = pd.DataFrame(stability_cells)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharey=True)
    for ax, stratum in zip(axes, STRATA):
        subset = stability[stability["stratum"] == stratum]
        data = [subset.loc[subset["requested_model"] == model, "mean_pairwise_jaccard"].to_numpy() for model in EXPECTED_MODELS]
        box = ax.boxplot(
            data, tick_labels=["5.6 Luna", "6 Luna"], patch_artist=True, widths=0.55, showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 4},
        )
        for patch, model in zip(box["boxes"], EXPECTED_MODELS):
            patch.set_facecolor(COLORS[model]); patch.set_alpha(0.75)
        for index, values in enumerate(data, 1):
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.asarray([])
            ax.scatter(np.full(len(values), index) + jitter, values, s=9, color="#333333", alpha=0.5, zorder=3)
        ax.set_ylim(0, 1.02)
        ax.set_title(STRATUM_LABELS[stratum])
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_ylabel("Mean within-cell pairwise Jaccard")
    fig.suptitle("Run-to-run model-produced event-set stability, separated by stratum")
    fig.tight_layout()
    save_figure(fig, output_dir, "generation_stability")
    generated.extend(["generation_stability.svg", "generation_stability.png"])

    bins = pd.DataFrame(calibration_bins)
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8), sharex=True, sharey=True)
    for ax, stratum in zip(axes, STRATA):
        plotted = bins[(bins["stratum"] == stratum) & (bins["n"] > 0)]
        ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
        sizes = 20 + 2 * plotted["n"].to_numpy(dtype=float)
        ax.scatter(
            plotted["mean_confidence"], plotted["observed_correct_fraction"], s=sizes,
            color="#0072B2", edgecolor="white", linewidth=0.7,
        )
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean stated confidence", title=STRATUM_LABELS[stratum])
        ax.grid(color="#eeeeee", linewidth=0.6)
    axes[0].set_ylabel("Observed identity correctness")
    fig.suptitle("Confidence calibration for emitted events, separated by stratum")
    fig.tight_layout()
    save_figure(fig, output_dir, "confidence_calibration")
    generated.extend(["confidence_calibration.svg", "confidence_calibration.png"])

    exact = cost["exact_same_token_counterfactual"]["arms"]
    labels = ["5.6 Sol\nsame-token", "6 Sol\nsame-token", "5.6 Sol\nlow plan", "6 Sol\nlow plan"]
    values = [exact["gpt-5.6-sol"]["standard_same_token_cost_usd"], exact["gpt-6-sol"]["standard_same_token_cost_usd"], cost["bounded_scenarios"]["gpt-5.6-sol"]["conservative_planning_low"]["cost_usd"], cost["bounded_scenarios"]["gpt-6-sol"]["conservative_planning_low"]["cost_usd"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(np.arange(4), values, color=["#0072B2", "#D55E00", "#56B4E9", "#E69F00"], width=0.68)
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Estimated Standard-processing cost (USD)")
    ax.set_title("Counterfactual Sol cost scenarios for matched 111-request arms")
    for index, value in enumerate(values):
        ax.text(index, value, f"${value:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output_dir, "cost_scenarios")
    generated.extend(["cost_scenarios.svg", "cost_scenarios.png"])
    return generated


def metric_row(summary_rows: Sequence[dict[str, Any]], model: str, tier: str, event_class: str, metric: str) -> dict[str, Any]:
    matches = [row for row in summary_rows if row["requested_model"] == model and row["complexity_tier"] == tier and row["event_class"] == event_class and row["metric"] == metric]
    if len(matches) != 1:
        raise AnalysisError(f"Expected one summary row for {(model, tier, event_class, metric)}, found {len(matches)}")
    return matches[0]


def fmt_prop(value: Any) -> str:
    return "NA" if value is None else f"{100 * float(value):.1f}%"


def fmt_ci(row: dict[str, Any]) -> str:
    return f"{fmt_prop(row['estimate'])} (95% CI {fmt_prop(row['ci95_percentile_low'])}–{fmt_prop(row['ci95_percentile_high'])})"


def fmt_p(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6g}"


def create_report(
    output_dir: Path,
    summary_rows: Sequence[dict[str, Any]],
    regression: dict[str, Any],
    response_metrics: Sequence[dict[str, Any]],
    temperature_rows: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
    stability: dict[str, Any],
    polymer: dict[str, Any],
    cocoa: dict[str, Any],
    sensitivities: dict[str, Any],
    costs: dict[str, Any],
    reconciliation_rows: Sequence[dict[str, Any]],
) -> str:
    complexity = {tier: metric_row(summary_rows, "all", tier, "all", "recall") for tier in ("M1", "M2", "M3")}
    model_rows = {model: metric_row(summary_rows, model, "all", "all", "recall") for model in EXPECTED_MODELS}
    overall = metric_row(summary_rows, "all", "all", "all", "recall")
    precision = metric_row(summary_rows, "all", "all", "all", "precision")
    f1 = metric_row(summary_rows, "all", "all", "all", "f1")
    terminal = metric_row(summary_rows, "all", "all", "terminal_melting", "recall")
    nonmelting = metric_row(summary_rows, "all", "all", "non_melting_pathway", "recall")
    direction = metric_row(summary_rows, "all", "all", "all", "direction_accuracy")
    terminal_direction = metric_row(summary_rows, "all", "all", "terminal_melting", "direction_accuracy")
    nonmelting_direction = metric_row(summary_rows, "all", "all", "non_melting_pathway", "direction_accuracy")
    incomplete = [row for row in response_metrics if row["completion_state"] == "incomplete"]
    temp_overall = next(row for row in temperature_rows if row["event_class"] == "all" and row["temperature_field"] == "all")
    regression_test = regression["joint_primary_M1_M3_complexity_test"]
    bca_rows = sensitivities["7_BCa_cluster_bootstrap"]["rows"]
    bca_unstable = [row for row in bca_rows if not row["stable"]]
    exact_cost = costs["exact_same_token_counterfactual"]
    cap = costs["output_ceiling_analysis"]["recommended_future_low_effort_fail_closed_cap_tokens"]
    visible_caps = costs["output_ceiling_analysis"]["completed_nonreasoning_proxy_caps_tokens"]
    reasoning_allowances = costs["output_ceiling_analysis"]["reasoning_allowances_tokens"]
    partial = sensitivities["1_partial_ambiguous_half_credit"]
    parseable = sensitivities["10_parseable_outputs_only"]
    glmm_max_abs_beta = max(abs(row["estimate_log_odds"]) for row in regression["mixed_model_attempt"]["coefficients"])
    bca_statement = "BCa intervals were numerically stable for all three complexity classes." if not bca_unstable else "BCa was numerically unstable for " + ", ".join(row["complexity_tier"] for row in bca_unstable) + "; percentile intervals remain primary."

    polymer_lines = []
    for row in polymer["by_tier"]:
        polymer_lines.append(f"- {row['complexity_tier']}: {row['material_count']} materials; identity recall {fmt_prop(row['recall'])} ({row['correct_identities']}/{row['gold_event_opportunities']} gold-event opportunities across arms and generations).")
    cocoa_lines = []
    for row in cocoa["rows"]:
        cocoa_lines.append(f"- `{row['requested_model']}`: recall {fmt_prop(row['recall'])} ({row['correct_identities']}/{row['gold_event_opportunities']}), precision {fmt_prop(row['precision'])}, F1 {row['f1']:.3f} across {row['responses']} generations.")
    incomplete_lines = [f"- `{row['key']}`: `{row['returned_model']}`, status `incomplete`, reason `max_output_tokens`; retained in the intention-to-benchmark denominator with zero recovered events." for row in incomplete]

    def calibration_stability_line(stratum: str) -> str:
        cal = calibration[stratum]
        stable = stability[stratum]
        return (
            f"Calibration conditional on emission used n = {cal['n_emitted']} events "
            f"(Brier {cal['brier_score']:.3f}; 10-bin ECE {cal['expected_calibration_error_10_equal_width_bins']:.3f}). "
            f"Across {stable['cell_count']} model–material cells, mean pairwise Jaccard was "
            f"{stable['material_first_summary']['mean_pairwise_jaccard']['mean']:.3f}, gold-identity unanimity was "
            f"{fmt_prop(stable['gold_identity_unanimity_rate'])}, and parse/failure-status concordance was "
            f"{fmt_prop(stable['material_first_summary']['parse_failure_status_concordance_rate'])}."
        )

    report = f"""# Frozen DSC benchmark: corrected results and Sol cost analysis

## Primary scientific result

Descriptive identity recovery fell across the prespecified molecular complexity classes: M1 {fmt_ci(complexity['M1'])} ({complexity['M1']['true_positive_numerator']}/{complexity['M1']['gold_denominator']}), M2 {fmt_ci(complexity['M2'])} ({complexity['M2']['true_positive_numerator']}/{complexity['M2']['gold_denominator']}), and M3 {fmt_ci(complexity['M3'])} ({complexity['M3']['true_positive_numerator']}/{complexity['M3']['gold_denominator']}). The frozen primary inferential test was not estimable. The random-intercept GLMM separated (maximum |β| = {glmm_max_abs_beta:.3f}), and the sole permitted exchangeable material-cluster GEE fallback also separated and produced an invalid robust covariance (minimum eigenvalue {regression['GEE_fallback']['robust_covariance_minimum_eigenvalue']:.3f}). The primary χ² statistic and p-value are therefore recorded as null, not replaced by another test. The marginal cluster-bootstrap contrasts are retained only as unprespecified exploratory estimates, with percentile intervals and Holm arithmetic, in [`primary_results.json`](primary_results.json). Interpretation is confined to descriptive recovery and its uncertainty; model-equivalence and global-ranking conclusions are outside the estimable result.

Across the 30-compound molecular cohort, intention-to-benchmark recall was {fmt_ci(overall)} ({overall['true_positive_numerator']}/{overall['gold_denominator']}), precision was {fmt_ci(precision)} ({precision['true_positive_numerator']}/{precision['predicted_denominator']}), and F1 was {fmt_ci(f1)}. Arm-specific recall was `{EXPECTED_MODELS[0]}` {fmt_ci(model_rows[EXPECTED_MODELS[0]])} and `{EXPECTED_MODELS[1]}` {fmt_ci(model_rows[EXPECTED_MODELS[1]])}.

## Completion and identity scoring

The audit validated all 222 unique planned/attempt/response keys, three generations per model–material cell, every prompt hash, exact requested/returned model identity, and the manifest totals before scoring. Of 222 outcomes, 219 were completed and parseable and three reached the frozen 4,000-token limit:

{chr(10).join(incomplete_lines)}

The parser retained these as distinct incomplete, unparseable outcomes and did not salvage truncated JSON. Raw collection files remained byte-identical. The temperature-blind one-to-one matcher applied the conservative identity rule. Partial identities counted as both a missed gold event and an unmatched prediction; half-credit sensitivity recall was {fmt_prop(partial['recall'])} and precision was {fmt_prop(partial['precision'])}.

The prior second review covered all 72 adjudication rows. For {len(reconciliation_rows)} disputed partial rows, the conservative partial label was retained while the exact gold-unit assignment was marked unresolved. Those assignments are blank in scored event records and excluded from secondary identity and stability calculations. [`adjudication_reconciliation.csv`](adjudication_reconciliation.csv) records the first-review assignment, reconciled label, unresolved status, and absence of consensus; [`blinded_adjudication.csv`](blinded_adjudication.csv) remains free of model labels, confidence, and temperature error.

## Molecular secondary outcomes

Terminal-melt recall was {fmt_ci(terminal)} ({terminal['true_positive_numerator']}/{terminal['gold_denominator']}), compared with {fmt_ci(nonmelting)} ({nonmelting['true_positive_numerator']}/{nonmelting['gold_denominator']}) for non-melting pathway events. Among identity-correct events, direction was correct for {terminal_direction['metric_numerator']}/{terminal_direction['metric_denominator']} terminal melts ({fmt_ci(terminal_direction)}) and {nonmelting_direction['metric_numerator']}/{nonmelting_direction['metric_denominator']} non-melting pathway events ({fmt_ci(nonmelting_direction)}); overall direction accuracy was {direction['metric_numerator']}/{direction['metric_denominator']} ({fmt_ci(direction)}). This decomposition preserves the shared-terminal-melt rule.

Temperature evidence was sparse: {temp_overall['identity_and_field_compatible_n']} identity-correct events from {temp_overall['material_n']} molecular material had compatible fields. Their median absolute error was {temp_overall['median_absolute_error_C']:.2f} °C (IQR {temp_overall['iqr_low_C']:.2f}–{temp_overall['iqr_high_C']:.2f} °C), and the material-level mean was {temp_overall['material_level_mean_absolute_error_C']:.2f} °C (95% cluster-bootstrap CI {temp_overall['material_level_mean_ci95_percentile_low_C']:.2f}–{temp_overall['material_level_mean_ci95_percentile_high_C']:.2f} °C). This conditional estimate describes only that observed material/event set. Direction-aware one-sided bounds and explicit handling for every gold temperature kind are implemented; [`temperature_summary.csv`](temperature_summary.csv) reports the field-specific results.

{calibration_stability_line('molecular_polymorph')} Calibration and model-produced event-set stability are conditional secondary outcomes and use molecular records only in this section.

## Polymer ladder and cocoa-butter case

The polymer controls remain a separate descriptive ladder:

{chr(10).join(polymer_lines)}

The single Ivory Coast cocoa-butter case remains a multicomponent polymorphic material and is excluded from pure-compound inference:

{chr(10).join(cocoa_lines)}

Material/model values underlying both summaries are in [`stratum_results.csv`](stratum_results.csv).

{calibration_stability_line('polymer_control')}

{calibration_stability_line('complex_material')} Calibration and stability for these strata are never pooled with molecular results or with each other. The stratum-specific bins, cells, and details are in [`calibration_bins.csv`](calibration_bins.csv), [`stability_cells.csv`](stability_cells.csv), and [`stability_results.json`](stability_results.json).

## Prespecified sensitivities

The prespecified GEE comparison was not estimable for robust inference because separation produced an indefinite sandwich covariance; its fit diagnostics are preserved. Protocol-complete, point-anchor-only, equal-material-weight, parseable-only, identifier-integrity, field-agnostic-temperature, half-credit, leave-one-material-out, BCa, and GEE records are in [`sensitivity_results.json`](sensitivity_results.json). The largest leave-one-material-out recall change was {sensitivities['6_leave_one_material_out']['largest_absolute_change']['change_from_full']:+.4f} after omitting `{sensitivities['6_leave_one_material_out']['largest_absolute_change']['material_id']}`. Parseable-only recall was {fmt_prop(parseable['recall'])} after excluding {parseable['excluded_response_count']} incomplete responses; this selection-prone estimate remains secondary to the intention-to-benchmark result. {bca_statement}

## Observed usage and counterfactual Sol cost

Observed Luna usage totaled {int(costs['observed_facts']['input_tokens']['sum']):,} input tokens, {int(costs['observed_facts']['reasoning_tokens']['sum']):,} reasoning tokens, and {int(costs['observed_facts']['total_output_tokens']['sum']):,} total output tokens across 222 requests, for a frozen-rate observed cost of `${costs['observed_facts']['luna_cost_usd']:.7f}`. Subtracting itemized reasoning from total output gives a usage-derived nonreasoning-output proxy of {int(costs['observed_facts']['usage_derived_nonreasoning_output_tokens']['sum']):,} tokens; formatting tokens may be unitemized. Distributions overall and by model, complexity, and completion state are in [`token_distributions.csv`](token_distributions.csv).

The Sol figures are separate same-token counterfactuals: `${exact_cost['arms']['gpt-5.6-sol']['standard_same_token_cost_usd']:.6f}` for the matched 111-request `gpt-5.6-sol` arm, `${exact_cost['arms']['gpt-6-sol']['standard_same_token_cost_usd']:.6f}` for the matched 111-request `gpt-6-sol` arm, `${exact_cost['two_arm_222_request_standard_cost_usd']:.6f}` for both arms under Standard processing, and `${exact_cost['two_arm_batch_cost_usd']:.6f}` under the eligible Batch/Flex half-rate assumption. No Sol usage was observed or predicted. Official model pages list `gpt-5.6-sol` at $4/M input and $20/M output and `gpt-6-sol` at $2/M input and $10/M output; both expose the documented reasoning controls. [GPT-5.6 Sol model page]({OFFICIAL_DOCS['gpt-5.6-sol']}), [GPT-6 Sol model page]({OFFICIAL_DOCS['gpt-6-sol']}), [OpenAI API pricing]({OFFICIAL_DOCS['pricing']}).

Among completed responses, the usage-derived nonreasoning-output proxy required at most {visible_caps['p95']:,} tokens at the 95th percentile, {visible_caps['p99']:,} at the 99th percentile, and {visible_caps['max']:,} at the observed maximum after rounding upward to 128-token increments. These are visible-payload planning floors, not safe total caps. OpenAI documents that `max_output_tokens` includes reasoning, visible output, and non-visible formatting tokens; a response can exhaust the limit before completing visible JSON. [Reasoning guide]({OFFICIAL_DOCS['reasoning']}), [token-counting guide]({OFFICIAL_DOCS['token_counting']}). The bounded low-effort planning allowance uses the observed-medium median reasoning load ({reasoning_allowances['low_planning_allowance']:,} tokens) as an explicit, unmeasured allowance; the corresponding arm-level scenarios and hard reservations are in [`cost_scenarios.json`](cost_scenarios.json).

## Sol-run recommendation and remaining gates

The cost-minimizing next step is an authorization-gated 18-request settings pilot before any 222-request Sol comparison: three fixed molecular materials per M1–M3 tier, one request per exact Sol arm, the frozen prompt/schema, `reasoning.effort='low'`, and fail-closed `max_output_tokens={cap}`. Use Batch or Flex only when eligible, judge the pilot on completion, schema validity, and token use, and proceed to a full comparison only after explicit API and cost-cap authorization.
"""
    atomic_write_text(output_dir / "RESULTS_REPORT.md", report)
    return report


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": sm.__version__,
        "matplotlib": matplotlib.__version__,
    }


def run_full_analysis(root: Path, output_dir: Path, bootstrap_reps: int, seed: int) -> dict[str, Any]:
    if bootstrap_reps != 10_000:
        raise AnalysisError("The frozen full analysis requires exactly 10,000 bootstrap resamples")
    manifest, materials, gold, plan, attempts, responses, input_hashes = load_analysis_inputs(root)
    materials_by_id = {row["material_id"]: row for row in materials["materials"]}
    gold_units = make_gold_units(gold, materials_by_id)
    decisions_path = root / "analysis" / "adjudication_decisions.json"
    decisions = load_decisions(decisions_path)
    parsed, event_rows, adjudication_rows, _ = parse_and_match(responses, gold_units, decisions)
    adjudication_ids = {row["adjudication_id"] for row in adjudication_rows}
    if set(decisions) != adjudication_ids:
        missing = sorted(adjudication_ids - set(decisions))
        extra = sorted(set(decisions) - adjudication_ids)
        raise AnalysisError(f"Adjudication decision coverage mismatch; missing={missing}, extra={extra}")
    for row in adjudication_rows:
        if row["rationale"].startswith("Conservative first-review"):
            raise AnalysisError(f"Unreviewed adjudication remains: {row['adjudication_id']}")
    attach_conditional_scores(event_rows, gold_units)
    for key, rows in pd.DataFrame(event_rows).query("row_type == 'predicted_event' and final_label == 'correct_identity'").groupby("key"):
        identifiers = [value for value in rows["gold_scoring_unit_id"].tolist() if value]
        if len(identifiers) != len(set(identifiers)):
            raise AnalysisError(f"Adjudication produced duplicate correct assignment in {key}")

    response_metrics = create_response_metrics(parsed, event_rows, gold_units, materials_by_id)
    gold_outcomes = make_gold_outcomes(response_metrics, event_rows, gold_units)
    summary_rows, bootstrap_distributions, material_ids, weights = summary_metrics_with_bootstrap(response_metrics, bootstrap_reps, seed)
    regression, gee = model_inference(gold_outcomes, bootstrap_distributions)
    calibration, calibration_bins = calibration_analysis(event_rows, response_metrics)
    stability, stability_cells, stability_detail = stability_analysis(response_metrics, event_rows, gold_units)
    temperature_rows = conditional_temperature_summary(event_rows, material_ids, weights)
    stratum_rows, polymer, cocoa = descriptive_strata(response_metrics)
    sensitivities = sensitivity_analyses(response_metrics, event_rows, gold_units, gold_outcomes, gold["protocols"], bootstrap_distributions, gee)
    costs = cost_analysis(response_metrics)
    token_rows = token_distribution_rows(response_metrics)
    reconciliation_rows = adjudication_reconciliation_rows(adjudication_rows)
    if len(reconciliation_rows) != 6:
        raise AnalysisError(f"Expected six unresolved second-review gold-unit assignments, found {len(reconciliation_rows)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "response_metrics.csv", response_metrics)
    write_csv(output_dir / "event_matches.csv", event_rows, EVENT_MATCH_FIELDS)
    write_csv(output_dir / "blinded_adjudication.csv", adjudication_rows, ADJUDICATION_FIELDS)
    write_csv(output_dir / "adjudication_reconciliation.csv", reconciliation_rows, RECONCILIATION_FIELDS)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    write_csv(output_dir / "gold_outcomes.csv", gold_outcomes)
    write_csv(output_dir / "temperature_summary.csv", temperature_rows)
    write_csv(output_dir / "calibration_bins.csv", calibration_bins)
    write_csv(output_dir / "stability_cells.csv", stability_cells)
    write_csv(output_dir / "stability_detail.csv", stability_detail)
    write_csv(output_dir / "stratum_results.csv", stratum_rows)
    write_csv(output_dir / "token_distributions.csv", token_rows)

    primary_results = {
        "study_id": STUDY_ID,
        "analysis_population": "30 molecular polymorph materials; intention-to-benchmark",
        "validated_collection": {
            "planned_response_attempt_keys": EXPECTED_REQUESTS, "unique_keys": EXPECTED_REQUESTS,
            "completed": EXPECTED_COMPLETED, "incomplete": EXPECTED_INCOMPLETE,
            "requested_returned_model_mismatches": 0, "generations_per_model_material_cell": 3,
            "raw_embedded_response_objects_validated": EXPECTED_REQUESTS,
        },
        "gold_scoring_units": {"all_strata": len(gold_units), "primary_molecular": sum(unit["counts_toward_primary"] for unit in gold_units)},
        "marginal_metrics": summary_rows, "cluster_aware_binary_model": regression,
        "conditional_temperature": temperature_rows, "confidence_calibration_by_stratum": calibration,
        "adjudication_reconciliation": {
            "unresolved_gold_unit_assignments": len(reconciliation_rows),
            "reconciled_label": "partial_ambiguous",
            "used_in_secondary_identity_or_stability": False,
            "table": "adjudication_reconciliation.csv",
        },
        "polymer_ladder": polymer, "cocoa_butter_case": cocoa,
        "incomplete_outcomes": [{key: row[key] for key in ("key", "material_id", "requested_model", "returned_model", "generation", "completion_state", "incomplete_reason", "parse_status")} for row in response_metrics if row["completion_state"] == "incomplete"],
    }
    write_json(output_dir / "primary_results.json", primary_results)
    write_json(output_dir / "sensitivity_results.json", sensitivities)
    write_json(output_dir / "stability_results.json", {"summary": stability, "cells_table": "stability_cells.csv", "detail_table": "stability_detail.csv"})
    write_json(output_dir / "cost_scenarios.json", costs)
    figure_files = create_figures(output_dir, summary_rows, stability_cells, calibration_bins, costs)
    create_report(
        output_dir, summary_rows, regression, response_metrics, temperature_rows, calibration, stability,
        polymer, cocoa, sensitivities, costs, reconciliation_rows,
    )

    after_hashes = {
        "manifest": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "MANIFEST.json"),
        "plan": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "execution-plan.jsonl"),
        "attempts": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "attempt-ledger.jsonl"),
        "responses": sha256_file(root / "outputs" / "luna-api-run-2026-09-27" / "response-ledger.jsonl"),
        "gold": sha256_file(root / "cohort" / "literature_reference.json"),
        "materials": sha256_file(root / "cohort" / "materials.json"),
        "packet": sha256_file(root / "blind" / "MODEL_PACKET.json"),
    }
    if input_hashes != after_hashes:
        raise AnalysisError("Raw/frozen inputs changed during analysis")
    required = [
        "adjudication_preparation_summary.json", "adjudication_reconciliation.csv", "analysis_manifest.json",
        "blinded_adjudication.csv", "calibration_bins.csv", "confidence_calibration.png", "confidence_calibration.svg",
        "cost_scenarios.json", "cost_scenarios.png", "cost_scenarios.svg", "event_class_recovery.png",
        "event_class_recovery.svg", "event_matches.csv", "generation_stability.png", "generation_stability.svg",
        "gold_outcomes.csv", "primary_complexity_model.png", "primary_complexity_model.svg", "primary_results.json",
        "response_metrics.csv", "RESULTS_REPORT.md", "sensitivity_results.json", "stability_cells.csv",
        "stability_detail.csv", "stability_results.json", "stratum_results.csv", "summary_metrics.csv",
        "temperature_summary.csv", "token_distributions.csv",
    ]
    missing = [name for name in required if name != "analysis_manifest.json" and (not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0)]
    if missing:
        raise AnalysisError(f"Required outputs missing or empty: {missing}")
    artifact_hashes = {path.name: sha256_file(path) for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "analysis_manifest.json"}
    analysis_manifest = {
        "study_id": STUDY_ID, "analysis_status": "PASS", "generated_at_utc": ANALYSIS_TIMESTAMP_UTC,
        "input_sha256_before": input_hashes, "input_sha256_after": after_hashes, "raw_inputs_preserved": input_hashes == after_hashes,
        "analysis_inputs_sha256": {
            "analysis_plan": sha256_file(root / "ANALYSIS_PLAN.md"),
            "study_design_freeze": sha256_file(root / "STUDY_DESIGN_FREEZE.md"),
            "protocol": sha256_file(root / "PROTOCOL.md"),
            "analysis_code": sha256_file(root / "analysis" / "analyze_dsc_benchmark.py"),
            "analysis_tests": sha256_file(root / "analysis" / "test_analyze_dsc_benchmark.py"),
            "adjudication_decisions": sha256_file(decisions_path),
        },
        "validated_counts": {"requests": len(responses), "completed": EXPECTED_COMPLETED, "incomplete": EXPECTED_INCOMPLETE, "parsed_completed": EXPECTED_COMPLETED, "gold_units_all": len(gold_units), "gold_units_primary": 89, "adjudication_rows": len(adjudication_rows)},
        "bootstrap": {"resamples": bootstrap_reps, "seed": seed, "stratification": "molecular complexity class", "cluster": "material", "primary_interval": "percentile"},
        "software_versions": software_versions(),
        "commands": [
            "analysis/.venv/bin/python analysis/analyze_dsc_benchmark.py --prepare-adjudication",
            "analysis/.venv/bin/python analysis/analyze_dsc_benchmark.py",
            "analysis/.venv/bin/python -m pytest -q analysis/test_analyze_dsc_benchmark.py",
        ],
        "official_documentation_checked_on": "2026-09-27", "official_documentation": OFFICIAL_DOCS,
        "artifacts_sha256": artifact_hashes, "required_outputs": required, "figure_files": figure_files,
        "deterministic_regeneration": {
            "fixed_analysis_timestamp_utc": ANALYSIS_TIMESTAMP_UTC,
            "svg_hash_salt": STUDY_ID,
            "verification": "pytest test_deterministic_double_regeneration runs two clean prepare+full regenerations and compares every file hash",
        },
        "adjudication": {
            "first_review_complete": True, "second_review_incorporated": True, "model_blinded": True,
            "temperature_error_blinded": True, "unresolved_gold_unit_assignments": len(reconciliation_rows),
            "unresolved_assignments_used_in_secondary_identity_or_stability": False, "consensus_claimed": False,
        },
    }
    write_json(output_dir / "analysis_manifest.json", analysis_manifest)
    return analysis_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_root / "results" / "dsc-analysis-2026-09-27")
    parser.add_argument("--prepare-adjudication", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    if args.prepare_adjudication:
        summary = prepare_adjudication(root, output_dir)
        print(json.dumps(summary, indent=2))
        return 0
    summary = run_full_analysis(root, output_dir, args.bootstrap_reps, args.seed)
    print(json.dumps({"analysis_status": summary["analysis_status"], "validated_counts": summary["validated_counts"], "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(f"ANALYSIS_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
