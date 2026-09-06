#!/usr/bin/env python3
"""V2-A Pareto: Task 1 — Candidate Canonicalization (spec section 5.2).

Reads the frozen 24,869 source candidates from projection.jsonl (legacy v1)
and native-candidates.jsonl (v2-vh), canonicalizes each into exactly
1 HyDE + 1–2 Vec, and writes the canonicalization ledger.

Canonicalization is DELETE-ONLY: it removes whole Lex lines, exact-duplicate
Vec lines, and excess Vec lines beyond the cap.  It never truncates words,
characters, or tokens within a HyDE/Vec line.  Candidates that exceed the
hard length caps are rejected as contract_invalid.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataset.scifact_distill import load_tokenizer

# === Frozen paths (spec section 5) ======================================

_SCRIPT_DIR = Path(__file__).resolve().parent.parent  # finetune/

PROJECTION_PATH = _SCRIPT_DIR / "data/public-distill-v0/experiments/v2-vh/projection.jsonl"
NATIVE_PATH = _SCRIPT_DIR / "data/public-distill-v0/experiments/v2-vh/native-candidates.jsonl"
OUTPUT_DIR = _SCRIPT_DIR / (
    "data/public-distill-v3-vh-pareto/experiments/"
    "public-main-v3-vh-pareto-prompt-v1/1-canonicalization"
)

# === Frozen tokenizer identity ==========================================

TOKENIZER_MODEL = "Qwen/Qwen3-1.7B"
TOKENIZER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# === Contract hard limits (spec section 6) ==============================

HYDE_WORD_HARD_CAP = 200  # words, not tokens
VEC_TOKEN_HARD_CAP = 48   # pinned Qwen tokens
COMPLETION_TOKEN_HARD_CAP = 384  # pinned Qwen tokens
VEC_COUNT_CAP = 2

# === Helpers ============================================================


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
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


# === Canonicalization ===================================================


@dataclass
class DeletionSummary:
    dropped_lex: int = 0
    dropped_duplicate_vec: list[int] = field(default_factory=list)
    dropped_excess_vec: list[int] = field(default_factory=list)


@dataclass
class CanonicalRecord:
    source_candidate_id: str
    canonical_candidate_id: str
    source_artifact: str  # "projection" or "native"
    source_candidate_index: int
    input_id: str
    qid: str
    query: str
    source_id: str
    provenance: dict[str, Any]
    raw_output: list[list]  # parsed [[type, text], ...]
    canonical_output: list[list] | None  # None if contract_invalid
    deletion: DeletionSummary
    contract_status: str  # "contract_valid" | "contract_invalid"
    contract_errors: list[str]
    failure_stage: str  # "canonicalization" | "contract_validation" | ""
    hyde_word_count: int | None
    vec_token_counts: list[int] | None
    completion_token_count: int | None


def _parse_raw_output(source_artifact: str, candidate: dict[str, Any]) -> list[list]:
    """Extract the raw parsed output from a source candidate."""
    if source_artifact == "projection":
        provenance = candidate.get("provenance", {})
        raw = provenance.get("original_parsed_output")
        if not isinstance(raw, list):
            raise ValueError(f"{candidate.get('input_id')}: missing original_parsed_output")
        return raw
    else:
        raw = candidate.get("output")
        if not isinstance(raw, list):
            raise ValueError(f"{candidate.get('input_id')}: missing output")
        return raw


def canonicalize_one(
    source_artifact: str,
    candidate: dict[str, Any],
    token_counter,
) -> CanonicalRecord:
    """Canonicalize a single source candidate per spec section 5.2."""
    input_id = candidate["input_id"]
    source_candidate_id = f"{input_id}:{candidate['candidate_index']}"
    raw_output = _parse_raw_output(source_artifact, candidate)
    deletion = DeletionSummary()
    errors: list[str] = []
    failure_stage = ""

    # Step 1: strip whitespace from each payload, keep raw bytes
    stripped: list[tuple[int, str, str]] = []  # (orig_idx, type, text)
    for i, item in enumerate(raw_output):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            errors.append(f"output[{i}]: not a [type, text] pair")
            continue
        kind, text = item
        if not isinstance(kind, str) or not isinstance(text, str):
            errors.append(f"output[{i}]: type/text must be strings")
            continue
        stripped.append((i, kind, text.strip()))

    # Step 2: remove all lex lines
    after_lex_removal: list[tuple[int, str, str]] = []
    for orig_idx, kind, text in stripped:
        if kind == "lex":
            deletion.dropped_lex += 1
            continue
        after_lex_removal.append((orig_idx, kind, text))

    # Step 3: check for exactly one hyde, collect hyde + vecs
    hyde_lines = [(i, t) for i, k, t in after_lex_removal if k == "hyde"]
    vec_lines = [(i, t) for i, k, t in after_lex_removal if k == "vec"]
    unknown = [k for _, k, _ in after_lex_removal if k not in ("hyde", "vec")]

    if unknown:
        errors.append(f"unknown output types: {unknown}")
        failure_stage = "canonicalization"

    if len(hyde_lines) == 0:
        errors.append("missing hyde line")
        failure_stage = "canonicalization"
    elif len(hyde_lines) > 1:
        errors.append(f"multiple hyde lines ({len(hyde_lines)})")
        failure_stage = "canonicalization"

    # Step 4: deduplicate vecs by exact match (after strip)
    seen_vec_texts: set[str] = set()
    deduped_vecs: list[tuple[int, str]] = []
    for orig_idx, text in vec_lines:
        if text in seen_vec_texts:
            deletion.dropped_duplicate_vec.append(orig_idx)
            continue
        seen_vec_texts.add(text)
        deduped_vecs.append((orig_idx, text))

    # Step 5: cap vec at 2, keeping first occurrences
    excess_vecs = deduped_vecs[VEC_COUNT_CAP:]
    deduped_vecs = deduped_vecs[:VEC_COUNT_CAP]
    for orig_idx, _ in excess_vecs:
        deletion.dropped_excess_vec.append(orig_idx)

    if len(deduped_vecs) == 0:
        errors.append("no vec lines after canonicalization")
        failure_stage = "canonicalization"

    # If canonicalization itself failed, stop here
    if failure_stage == "canonicalization":
        return CanonicalRecord(
            source_candidate_id=source_candidate_id,
            canonical_candidate_id="",
            source_artifact=source_artifact,
            source_candidate_index=candidate["candidate_index"],
            input_id=input_id,
            qid=candidate.get("qid", ""),
            query=candidate.get("query", ""),
            source_id=candidate.get("source_id", ""),
            provenance=candidate.get("provenance", {}),
            raw_output=raw_output,
            canonical_output=None,
            deletion=deletion,
            contract_status="contract_invalid",
            contract_errors=errors,
            failure_stage="canonicalization",
            hyde_word_count=None,
            vec_token_counts=None,
            completion_token_count=None,
        )

    # Build canonical output: hyde, vec, vec (if present)
    hyde_text = hyde_lines[0][1]
    canonical_output: list[list] = [["hyde", hyde_text]]
    for _, vec_text in deduped_vecs:
        canonical_output.append(["vec", vec_text])

    # === Contract validation (spec section 6) ==========================

    # HyDE word count
    hyde_words = len(hyde_text.split())
    if hyde_words > HYDE_WORD_HARD_CAP:
        errors.append(
            f"hyde exceeds {HYDE_WORD_HARD_CAP} words ({hyde_words})"
        )
        failure_stage = "contract_validation"

    # Vec token counts
    vec_token_counts = [token_counter(v[1]) for v in canonical_output if v[0] == "vec"]
    for i, tc in enumerate(vec_token_counts):
        if tc > VEC_TOKEN_HARD_CAP:
            errors.append(
                f"vec[{i}] exceeds {VEC_TOKEN_HARD_CAP} tokens ({tc})"
            )
            failure_stage = failure_stage or "contract_validation"

    # Completion token count
    completion_text = "\n".join(
        f"{kind}: {text}" for kind, text in canonical_output
    )
    completion_tokens = token_counter(completion_text)
    if completion_tokens > COMPLETION_TOKEN_HARD_CAP:
        errors.append(
            f"completion exceeds {COMPLETION_TOKEN_HARD_CAP} tokens ({completion_tokens})"
        )
        failure_stage = failure_stage or "contract_validation"

    # Vec near-duplicate check (exact match after strip — already handled above,
    # but we also check case-insensitive normalized match)
    vec_texts = [v[1] for v in canonical_output if v[0] == "vec"]
    if len(vec_texts) == 2:
        nfkc0 = " ".join(vec_texts[0].casefold().split())
        nfkc1 = " ".join(vec_texts[1].casefold().split())
        if nfkc0 == nfkc1:
            errors.append("vec[0] and vec[1] are near-duplicates (NFKC casefold)")
            failure_stage = failure_stage or "contract_validation"

    # Vec must not be an exact copy of the input query
    query_nfkc = " ".join(candidate.get("query", "").casefold().split())
    for i, vt in enumerate(vec_texts):
        if " ".join(vt.casefold().split()) == query_nfkc:
            errors.append(f"vec[{i}] is an exact copy of the input query")
            failure_stage = failure_stage or "contract_validation"

    contract_status = "contract_valid" if not errors else "contract_invalid"

    # Generate canonical_candidate_id
    canonical_bytes = json.dumps(
        canonical_output, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    cid = sha256_hex(source_candidate_id.encode() + b"\0" + canonical_bytes)

    return CanonicalRecord(
        source_candidate_id=source_candidate_id,
        canonical_candidate_id=cid,
        source_artifact=source_artifact,
        source_candidate_index=candidate["candidate_index"],
        input_id=input_id,
        qid=candidate.get("qid", ""),
        query=candidate.get("query", ""),
        source_id=candidate.get("source_id", ""),
        provenance=candidate.get("provenance", {}),
        raw_output=raw_output,
        canonical_output=canonical_output,
        deletion=deletion,
        contract_status=contract_status,
        contract_errors=errors,
        failure_stage=failure_stage,
        hyde_word_count=hyde_words,
        vec_token_counts=vec_token_counts,
        completion_token_count=completion_tokens,
    )


# === Main ===============================================================


def run(projection_path: Path = PROJECTION_PATH,
        native_path: Path = NATIVE_PATH,
        output_dir: Path = OUTPUT_DIR) -> Path:
    """Run the full canonicalization pipeline; return the output directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_MODEL, TOKENIZER_REVISION, local_files_only=False)

    def token_counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Read source artifacts
    print(f"Reading projection.jsonl...")
    projection_records = read_jsonl(projection_path)
    print(f"  {len(projection_records)} records")

    print(f"Reading native-candidates.jsonl...")
    native_records = read_jsonl(native_path)
    print(f"  {len(native_records)} records")

    # Compute source artifact hashes
    projection_sha256 = sha256_hex(projection_path.read_bytes())
    native_sha256 = sha256_hex(native_path.read_bytes())

    # Canonicalize all candidates
    ledger: list[dict[str, Any]] = []
    canonical_candidates: list[dict[str, Any]] = []
    stats: dict[str, Counter] = {
        "contract_status": Counter(),
        "failure_stage": Counter(),
        "source_artifact": Counter(),
        "source_id": Counter(),
    }

    total = len(projection_records) + len(native_records)
    for source_artifact, records in [
        ("projection", projection_records),
        ("native", native_records),
    ]:
        for candidate in records:
            result = canonicalize_one(source_artifact, candidate, token_counter)

            # Ledger entry
            ledger_entry = {
                "source_candidate_id": result.source_candidate_id,
                "canonical_candidate_id": result.canonical_candidate_id,
                "source_artifact": result.source_artifact,
                "source_candidate_index": result.source_candidate_index,
                "input_id": result.input_id,
                "qid": result.qid,
                "source_id": result.source_id,
                "contract_status": result.contract_status,
                "failure_stage": result.failure_stage,
                "contract_errors": result.contract_errors,
                "deletion": {
                    "dropped_lex": result.deletion.dropped_lex,
                    "dropped_duplicate_vec": result.deletion.dropped_duplicate_vec,
                    "dropped_excess_vec": result.deletion.dropped_excess_vec,
                },
                "hyde_word_count": result.hyde_word_count,
                "vec_token_counts": result.vec_token_counts,
                "completion_token_count": result.completion_token_count,
            }
            ledger.append(ledger_entry)

            stats["contract_status"][result.contract_status] += 1
            stats["failure_stage"][result.failure_stage or "none"] += 1
            stats["source_artifact"][result.source_artifact] += 1
            stats["source_id"][result.source_id] += 1

            # Canonical candidates (only contract-valid ones go to retrieval)
            if result.contract_status == "contract_valid":
                canonical_candidates.append({
                    "source_candidate_id": result.source_candidate_id,
                    "canonical_candidate_id": result.canonical_candidate_id,
                    "input_id": result.input_id,
                    "qid": result.qid,
                    "query": result.query,
                    "source_id": result.source_id,
                    "source_artifact": result.source_artifact,
                    "canonical_output": result.canonical_output,
                })

    # Write outputs
    ledger_path = output_dir / "canonicalization-ledger.jsonl"
    atomic_jsonl(ledger_path, ledger)
    print(f"\nLedger: {len(ledger)} records → {ledger_path}")

    candidates_path = output_dir / "canonical-candidates.jsonl"
    atomic_jsonl(candidates_path, canonical_candidates)
    print(f"Canonical candidates: {len(canonical_candidates)} → {candidates_path}")

    # Unique query count
    unique_queries = len({c["input_id"] for c in canonical_candidates})
    print(f"Unique query groups: {unique_queries}")

    # Manifest
    manifest = {
        "schema_version": "qmd-v2a-canonicalization-v1",
        "source_artifacts": {
            "projection": {
                "path": str(projection_path),
                "records": len(projection_records),
                "sha256": projection_sha256,
            },
            "native": {
                "path": str(native_path),
                "records": len(native_records),
                "sha256": native_sha256,
            },
        },
        "total_source_candidates": total,
        "contract_valid_candidates": len(canonical_candidates),
        "unique_queries": unique_queries,
        "contract_status": dict(stats["contract_status"]),
        "failure_stage": dict(stats["failure_stage"]),
        "by_source_artifact": dict(stats["source_artifact"]),
        "by_source_id": dict(sorted(stats["source_id"].items())),
        "vec_count_distribution": dict(
            Counter(
                len([i for i in c["canonical_output"] if i[0] == "vec"])
                for c in canonical_candidates
            )
        ),
        "tokenizer": {
            "model": TOKENIZER_MODEL,
            "revision": TOKENIZER_REVISION,
        },
        "hard_limits": {
            "hyde_word_cap": HYDE_WORD_HARD_CAP,
            "vec_token_cap": VEC_TOKEN_HARD_CAP,
            "completion_token_cap": COMPLETION_TOKEN_HARD_CAP,
            "vec_count_cap": VEC_COUNT_CAP,
        },
    }
    manifest_path = output_dir / "canonicalization-manifest.json"
    atomic_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")

    return output_dir


if __name__ == "__main__":
    run()