#!/usr/bin/env node

import { createHash } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { dirname, join, resolve } from "node:path";

import { parseDistillRecords } from "../lib/distill.js";
import { truncateHydeAtSentenceBoundary } from "./hyde-truncation.js";

function sha256(text: string): string {
  return createHash("sha256").update(text).digest("hex");
}

function writeAtomic(path: string, text: string): void {
  const temporary = join(dirname(path), `.${path.split("/").at(-1)}.${process.pid}.tmp`);
  writeFileSync(temporary, text, "utf8");
  renameSync(temporary, path);
}

function usage(exitCode: number): never {
  const output = [
    "Usage:",
    "  node --import tsx finetune/benchmarks/distill/truncate-hyde.ts \\",
    "    --run-dir <unvalidated-derived-run> [--max-english-words 70]",
  ].join("\n");
  (exitCode === 0 ? console.log : console.error)(output);
  process.exit(exitCode);
}

const { values } = parseArgs({
  options: {
    "run-dir": { type: "string" },
    "max-english-words": { type: "string", default: "70" },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});
if (values.help) usage(0);
if (!values["run-dir"]) usage(1);

const maxWords = Number.parseInt(values["max-english-words"], 10);
if (!Number.isSafeInteger(maxWords) || maxWords <= 0) {
  throw new Error("--max-english-words must be a positive integer");
}

const runDir = resolve(values["run-dir"]);
const candidatesPath = join(runDir, "candidates.jsonl");
const manifestPath = join(runDir, "manifest.json");
if (!existsSync(candidatesPath) || !existsSync(manifestPath)) {
  throw new Error("run directory must contain candidates.jsonl and manifest.json");
}
const candidatesText = readFileSync(candidatesPath, "utf8");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Record<string, unknown>;
if (manifest.candidates_sha256 !== sha256(candidatesText)) {
  throw new Error("candidates.jsonl does not match the generated artifact hash");
}

const records = parseDistillRecords(candidatesText);
let totalHyde = 0;
let truncatedHyde = 0;
for (const record of records) {
  if (record.selection_status !== "pending" || record.raw_metrics !== null) {
    throw new Error(`qid ${record.qid}: derive this variant before Contract validation or scoring`);
  }
  for (const candidate of record.candidates) {
    candidate.parsed_output = candidate.parsed_output.map(([type, query]) => {
      if (type !== "hyde") return [type, query];
      totalHyde++;
      const truncated = truncateHydeAtSentenceBoundary(query, maxWords);
      if (truncated !== query) truncatedHyde++;
      return [type, truncated];
    });
    candidate.raw_output = JSON.stringify({
      expansions: candidate.parsed_output.map(([type, query]) => ({ type, query })),
    });
  }
}

const output = `${records.map(record => JSON.stringify(record)).join("\n")}\n`;
writeAtomic(candidatesPath, output);
manifest.candidates_sha256 = sha256(output);
manifest.derived_transform = {
  version: "hyde-sentence-prefix-max-english-words-v1",
  max_english_words: maxWords,
  source_candidates_sha256: sha256(candidatesText),
  total_hyde: totalHyde,
  truncated_hyde: truncatedHyde,
  unchanged_hyde: totalHyde - truncatedHyde,
};
manifest.validated_candidates_sha256 = null;
manifest.scored_candidates_sha256 = null;
manifest.selection_counts = null;
manifest.pilot_gate = null;
writeAtomic(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(JSON.stringify({
  run_dir: runDir,
  max_english_words: maxWords,
  total_hyde: totalHyde,
  truncated_hyde: truncatedHyde,
  unchanged_hyde: totalHyde - truncatedHyde,
}, null, 2));
