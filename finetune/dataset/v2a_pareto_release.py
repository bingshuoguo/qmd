#!/usr/bin/env python3
"""V2-A Pareto: Task 5 — Data Split & Release (spec sections 11-13).

Reads the selection-ledger and semantic-audit results, filters to
semantic-pass winners, applies the V2 student prompt, splits into
train/validation, and writes the sealed SFT release.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# === Paths ==============================================================

_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # finetune/
_EXPERIMENT_DIR = _SCRIPT_DIR / (
    "data/public-distill-v3-vh-pareto/experiments/"
    "public-main-v3-vh-pareto-prompt-v1"
)
_LEDGER_PATH = _EXPERIMENT_DIR / "3-pareto-selection" / "selection-ledger.jsonl"
_AUDIT_RAW_PATH = _EXPERIMENT_DIR / "4-semantic-audit" / "semantic-audit.raw.jsonl"
_CANONICAL_PATH = _EXPERIMENT_DIR / "1-canonicalization" / "canonical-candidates.jsonl"
_OUTPUT_DIR = _EXPERIMENT_DIR / "5-release"

# === V2-A Student Prompt (spec section 13) ==============================

V2A_PROMPT_TEMPLATE = (
    "/no_think You generate retrieval expansions for a search query.\n"
    "Return exactly one hyde line followed by one or two vec lines.\n"
    "Target 20-120 words for HyDE; use fewer for a simple query.\n"
    "Preserve entities, versions, numbers, constraints, negation, comparison, and intent.\n"
    "Do not invent unsupported facts, causal claims, laws, statistics, or definitions.\n"
    "Make Vec queries complementary rather than duplicates.\n"
    "Query: {query}"
)

PROMPT_VERSION = "qmd-student-expansion-v3-vh-pareto-v1"
RELEASE_ID = "public-distill-v3-vh-pareto"
EXPERIMENT_ID = "public-main-v3-vh-pareto-prompt-v1"

# === Frozen training config =============================================

MAX_SEQ_LENGTH = 1024

# === Helpers ============================================================


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def split_hash(input_id: str) -> str:
    return hashlib.sha256(
        f"qmd-v2a-pareto-split\0seed=42\0{input_id}".encode()
    ).hexdigest()


def render_prompt(query: str) -> str:
    return (
        "<|im_start|>user\n"
        f"{V2A_PROMPT_TEMPLATE.replace('{query}', query)}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


def render_completion(canonical_output: list[list]) -> str:
    return "\n".join(f"{kind}: {text}" for kind, text in canonical_output)


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if total < 0 or total > population:
        raise ValueError(f"invalid allocation total {total} for population {population}")
    allocation = {
        key: total * count // population for key, count in counts.items()
    }
    remaining = total - sum(allocation.values())
    order = sorted(
        counts, key=lambda key: (-(total * counts[key] % population), key)
    )
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


# === Main ===============================================================


def run(
    ledger_path: Path = _LEDGER_PATH,
    audit_raw_path: Path = _AUDIT_RAW_PATH,
    canonical_path: Path = _CANONICAL_PATH,
    output_dir: Path = _OUTPUT_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load semantic audit results
    print("Loading semantic audit...")
    audit_records = read_jsonl(audit_raw_path)
    semantic_pass: set[str] = set()
    semantic_status: dict[str, str] = {}
    for r in audit_records:
        sid = r["source_candidate_id"]
        semantic_status[sid] = r.get("overall", "unknown")
        if r.get("overall") == "pass":
            semantic_pass.add(sid)

    print(f"  {len(semantic_pass)} semantic-pass candidates")

    # Load canonical candidates for output text
    print("Loading canonical candidates...")
    canonical_records = read_jsonl(canonical_path)
    canonical_by_scid: dict[str, dict[str, Any]] = {}
    for c in canonical_records:
        canonical_by_scid[c["source_candidate_id"]] = c

    # Load selection ledger (winners only)
    print("Loading selection ledger...")
    ledger = read_jsonl(ledger_path)
    winners = [r for r in ledger if r.get("status") == "winner" and r.get("winner")]

    # Filter to semantic-pass winners
    accepted: list[dict[str, Any]] = []
    for w in winners:
        scid = w["winner"]["source_candidate_id"]
        if scid in semantic_pass:
            accepted.append(w)

    print(f"  {len(accepted)} accepted (winner + semantic-pass)")

    # Source distribution
    source_counts = Counter(w["source_id"] for w in accepted)
    print(f"  By source: {dict(source_counts)}")

    # Cap at 2,000 if needed
    if len(accepted) > 2000:
        print(f"  Capping from {len(accepted)} to 2000...")
        cap_allocation = largest_remainder(dict(source_counts), 2000)
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for w in accepted:
            by_source[w["source_id"]].append(w)
        accepted = []
        for src in sorted(by_source):
            ordered = sorted(
                by_source[src],
                key=lambda w: (
                    w["winner"]["canonical_candidate_id"],
                ),
            )
            accepted.extend(ordered[: cap_allocation[src]])
        print(f"  Capped to {len(accepted)}")

    if len(accepted) < 1000:
        raise ValueError(
            f"Only {len(accepted)} accepted records (< 1000 minimum). "
            "Training must not start. See spec section 11."
        )

    # Build SFT records
    print("Building SFT records...")
    sft_records: list[dict[str, Any]] = []
    for w in accepted:
        scid = w["winner"]["source_candidate_id"]
        canonical = canonical_by_scid.get(scid)
        if not canonical:
            raise ValueError(f"Canonical candidate not found: {scid}")

        prompt = render_prompt(canonical["query"])
        completion = render_completion(canonical["canonical_output"])

        sft_records.append({
            "schema_version": "qmd-public-distill-v3",
            "release_id": RELEASE_ID,
            "experiment_id": EXPERIMENT_ID,
            "input_id": w["input_id"],
            "qid": w["qid"],
            "query": canonical["query"],
            "source_id": w["source_id"],
            "source_candidate_id": scid,
            "canonical_candidate_id": w["winner"]["canonical_candidate_id"],
            "canonical_output": canonical["canonical_output"],
            "prompt": prompt,
            "completion": completion,
            "semantic_status": "pass",
            "final_sft_eligible": True,
            "smoke_only": False,
        })

    # Split into train/validation
    print("Splitting train/validation...")
    validation_target = max(1, len(sft_records) // 10)
    source_strata = Counter(r["source_id"] for r in sft_records)
    validation_allocation = largest_remainder(dict(source_strata), validation_target)

    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sft_records:
        by_stratum[r["source_id"]].append(r)

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for src in sorted(by_stratum):
        ordered = sorted(
            by_stratum[src],
            key=lambda r: (split_hash(r["input_id"]), r["input_id"]),
        )
        vcount = validation_allocation[src]
        validation.extend(
            {**r, "split": "validation"} for r in ordered[:vcount]
        )
        train.extend(
            {**r, "split": "train"} for r in ordered[vcount:]
        )

    combined = train + validation
    print(f"  Train: {len(train)}, Validation: {len(validation)}")

    # Write outputs
    combined_path = output_dir / "sft.jsonl"
    train_path = output_dir / "sft-train.jsonl"
    val_path = output_dir / "sft-validation.jsonl"

    atomic_jsonl(combined_path, combined)
    atomic_jsonl(train_path, train)
    atomic_jsonl(val_path, validation)

    # Prompt
    prompt_sha256 = sha256_hex(V2A_PROMPT_TEMPLATE.encode("utf-8"))

    # Release manifest
    release_root = output_dir.parent.parent

    def artifact(path: Path, rows: int) -> dict[str, Any]:
        return {
            "path": str(path.relative_to(release_root)),
            "bytes": path.stat().st_size,
            "rows": rows,
            "sha256": sha256_file(path),
        }

    release_manifest = {
        "schema_version": "qmd-public-distill-release-v3",
        "release_id": RELEASE_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "sealed",
        "final_sft_eligible": True,
        "provenance": {
            "pipeline_version": "v2a-pareto-v1",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_sha256,
            "prompt_template": V2A_PROMPT_TEMPLATE,
            "selection_rule": "strict_5_metric_pareto",
            "semantic_audit_version": "v2a-semantic-judge-v1",
        },
        "dataset": {
            "materialized": len(combined),
            "train": len(train),
            "validation": len(validation),
        },
        "core_artifacts": {
            "sft.jsonl": artifact(combined_path, len(combined)),
            "sft-train.jsonl": artifact(train_path, len(train)),
            "sft-validation.jsonl": artifact(val_path, len(validation)),
        },
        "mutation_policy": (
            "Any change to a sealed core artifact requires a new release_id "
            "and experiment_id."
        ),
    }

    manifest_path = output_dir / "release-manifest.json"
    atomic_json(manifest_path, release_manifest)

    # Source allocation report
    source_allocation = {
        "total_accepted": len(accepted),
        "total_materialized": len(combined),
        "train": len(train),
        "validation": len(validation),
        "by_source": {
            src: {
                "accepted": source_counts.get(src, 0),
                "train": sum(1 for r in train if r["source_id"] == src),
                "validation": sum(
                    1 for r in validation if r["source_id"] == src
                ),
            }
            for src in sorted(source_counts)
        },
    }
    atomic_json(output_dir / "source-allocation.json", source_allocation)

    print(f"\nRelease: {len(train)} train, {len(validation)} validation")
    print(f"Manifest: {manifest_path}")
    print(f"Prompt SHA-256: {prompt_sha256}")

    return output_dir


if __name__ == "__main__":
    run()