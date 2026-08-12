#!/usr/bin/env node

import { parseArgs } from "node:util";
import {
  generateExpansionArtifact,
  type ExpansionGenerationProgress,
} from "../../src/bench/expansions.js";
import { applyLlamaEnvMitigation } from "./lib/cli.js";

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run bench:expand -- --benchmark <dir> --run <name> --model <id-or-path> [--force]",
    "",
    "Examples:",
    "  npm run bench:expand -- --benchmark finetune/benchmarks/qmd-expansion-scifact-v1 \\",
    "    --run upstream-qmd --model hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf",
    "  npm run bench:expand -- --benchmark finetune/benchmarks/qmd-expansion-scifact-v1 \\",
    "    --run qwen-1.7b-base --model /absolute/path/to/base.gguf",
  ].join("\n");
  (exitCode === 0 ? console.log : console.error)(message);
  process.exit(exitCode);
}

function formatDuration(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function renderProgress(progress: ExpansionGenerationProgress): void {
  if (!process.stderr.isTTY) return;
  const width = 24;
  const ratio = progress.total === 0 ? 1 : progress.completed / progress.total;
  const filled = Math.round(ratio * width);
  const elapsed = formatDuration(progress.elapsed_ms);
  const eta = progress.completed === 0
    ? "--"
    : formatDuration((progress.elapsed_ms / progress.completed) * (progress.total - progress.completed));
  const last = progress.last_query_ms === null ? "--" : formatDuration(progress.last_query_ms);
  process.stderr.write(
    `\x1b[2K\rExpand [${"=".repeat(filled)}${"-".repeat(width - filled)}]`
    + ` ${Math.round(ratio * 100).toString().padStart(3)}%`
    + ` ${progress.completed}/${progress.total} elapsed=${elapsed} eta=${eta} last=${last}`
    + ` format=${progress.format_errors} generation=${progress.generation_errors}`
    + ` fallback=${progress.fallbacks}`
    + (progress.completed === progress.total ? "\n" : ""),
  );
}

const { values } = parseArgs({
  options: {
    benchmark: { type: "string" },
    run: { type: "string" },
    model: { type: "string" },
    force: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) usage(0);
if (!values.benchmark || !values.run || !values.model) usage(1);

applyLlamaEnvMitigation();

const { LlamaCpp } = await import("../../src/llm.js");
const llm = new LlamaCpp({
  generateModel: values.model,
  inactivityTimeoutMs: 0,
});

try {
  const summary = await generateExpansionArtifact({
    benchmarkDir: values.benchmark,
    runName: values.run,
    force: values.force,
    generateRaw: query => llm.generateQueryExpansionRaw(query),
    onProgress: renderProgress,
  });
  console.log(JSON.stringify({ model: values.model, ...summary }, null, 2));
} finally {
  await llm.dispose();
}
