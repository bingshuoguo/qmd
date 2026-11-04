#!/usr/bin/env node

import { createHash } from "node:crypto";
import {
  copyFileSync,
  existsSync,
  linkSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { join, resolve } from "node:path";

const PUBLIC_ROOT = resolve("finetune/data/public-distill-v0");
const PREPARED = join(PUBLIC_ROOT, "prepared");
const OUTPUT = join(PUBLIC_ROOT, "prepared-smoke-index");
const DOCUMENT_LIMIT = 2_000;
const SOURCE_IDS = ["fiqa-train", "cqadup-programmers", "cqadup-unix"];

function sha256(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function jsonl(path: string): any[] {
  return readFileSync(path, "utf8").trim().split("\n").filter(Boolean).map(line => JSON.parse(line));
}

function writeJsonl(path: string, values: readonly unknown[]): void {
  writeFileSync(path, `${values.map(value => JSON.stringify(value)).join("\n")}\n`, "utf8");
}

function writeJson(path: string, value: unknown): void {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

if (existsSync(OUTPUT)) throw new Error(`Smoke index preparation already exists: ${OUTPUT}`);
const staging = `${OUTPUT}.staging`;
rmSync(staging, { recursive: true, force: true });
mkdirSync(staging, { recursive: true });
try {
  const pool = jsonl(join(PREPARED, "pool-smoke.jsonl"));
  for (const sourceId of SOURCE_IDS) {
    const sourceRoot = join(PREPARED, sourceId);
    const outputRoot = join(staging, sourceId);
    mkdirSync(join(outputRoot, "corpus"), { recursive: true });
    const inputs = pool.filter(input => input.source_id === sourceId);
    if (inputs.length !== 10) throw new Error(`${sourceId}: expected 10 smoke inputs`);
    const qids = new Set(inputs.map(input => input.qid));
    const allQrels = readFileSync(join(sourceRoot, "qrels.tsv"), "utf8").trim().split(/\r?\n/u);
    const header = allQrels[0]!;
    const qrelLines = allQrels.slice(1).filter(line => qids.has(line.split("\t")[0]!));
    const positiveDocIds = new Set(qrelLines.map(line => line.split("\t")[1]!));
    const documents = jsonl(join(sourceRoot, "documents.jsonl"));
    const selected = documents
      .filter(document => positiveDocIds.has(document.doc_id))
      .concat(documents
        .filter(document => !positiveDocIds.has(document.doc_id))
        .sort((left, right) => {
          const leftKey = sha256(`qmd-public-smoke-index-v0\0${sourceId}\0${left.doc_id}`);
          const rightKey = sha256(`qmd-public-smoke-index-v0\0${sourceId}\0${right.doc_id}`);
          return leftKey.localeCompare(rightKey) || left.doc_id.localeCompare(right.doc_id);
        })
        .slice(0, DOCUMENT_LIMIT - positiveDocIds.size));
    if (selected.length !== DOCUMENT_LIMIT) {
      throw new Error(`${sourceId}: expected ${DOCUMENT_LIMIT} selected documents, got ${selected.length}`);
    }
    const selectedIds = new Set(selected.map(document => document.doc_id));
    for (const docId of positiveDocIds) {
      if (!selectedIds.has(docId)) throw new Error(`${sourceId}: missing positive document ${docId}`);
    }
    for (const document of selected) {
      linkSync(
        join(sourceRoot, "corpus", document.path),
        join(outputRoot, "corpus", document.path),
      );
    }
    writeJsonl(join(outputRoot, "queries.jsonl"), inputs.map(input => ({ qid: input.qid, query: input.query })));
    writeJsonl(join(outputRoot, "documents.jsonl"), selected);
    const reducedQrels = `${header}\n${qrelLines.join("\n")}\n`;
    writeFileSync(join(outputRoot, "qrels.tsv"), reducedQrels, "utf8");
    writeFileSync(join(outputRoot, "source-qrels.tsv"), reducedQrels, "utf8");
    writeJson(join(outputRoot, "excluded-qids.json"), []);
    for (const filename of ["retrieval-profile.yaml", "leakage-report.json"]) {
      copyFileSync(join(sourceRoot, filename), join(outputRoot, filename));
    }
    const sourceQrelsSha256 = sha256(readFileSync(join(outputRoot, "source-qrels.tsv")));
    const excludedQidsSha256 = sha256(readFileSync(join(outputRoot, "excluded-qids.json")));
    const leakageReportSha256 = sha256(readFileSync(join(outputRoot, "leakage-report.json")));
    const benchmark = readFileSync(join(sourceRoot, "benchmark.yaml"), "utf8")
      .replace(/^source_qrels_sha256: .+$/mu, `source_qrels_sha256: ${sourceQrelsSha256}`)
      .replace(/^excluded_qids_sha256: .+$/mu, `excluded_qids_sha256: ${excludedQidsSha256}`)
      .replace(/^leakage_report_sha256: .+$/mu, `leakage_report_sha256: ${leakageReportSha256}`);
    writeFileSync(join(outputRoot, "benchmark.yaml"), benchmark, "utf8");
    writeJson(join(outputRoot, "source-manifest.json"), {
      source_id: sourceId,
      schema_version: "qmd-public-smoke-reduced-index-v0",
      smoke_only: true,
      diagnostic_reduced_index: true,
      full_corpus_document_count: documents.length,
      indexed_document_count: selected.length,
      positive_document_count: positiveDocIds.size,
      deterministic_distractor_count: selected.length - positiveDocIds.size,
      selection: "all smoke qrels positives plus lowest sha256(qmd-public-smoke-index-v0\\0source_id\\0doc_id)",
      documents_sha256: sha256(readFileSync(join(outputRoot, "documents.jsonl"))),
      qrels_sha256: sha256(readFileSync(join(outputRoot, "qrels.tsv"))),
      queries_sha256: sha256(readFileSync(join(outputRoot, "queries.jsonl"))),
      warning: "Retrieval outcomes are diagnostic only and cannot select final SFT records.",
    });
  }
  writeJson(join(staging, "manifest.json"), {
    schema_version: "qmd-public-smoke-reduced-index-v0",
    smoke_only: true,
    diagnostic_reduced_index: true,
    documents_per_source: DOCUMENT_LIMIT,
    source_count: SOURCE_IDS.length,
    final_sft_eligible: false,
  });
  renameSync(staging, OUTPUT);
  process.stdout.write(`${JSON.stringify({ output: OUTPUT, documents_per_source: DOCUMENT_LIMIT }, null, 2)}\n`);
} catch (error) {
  rmSync(staging, { recursive: true, force: true });
  throw error;
}
