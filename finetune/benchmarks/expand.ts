#!/usr/bin/env node

import { parseArgs } from "node:util";
import {
  generateExpansionArtifact,
  fileSha256,
  type ExpansionGenerationProgress,
} from "../../src/bench/expansions.js";
import { loadInferenceProfile, inferenceProfileSha256 } from "../../src/bench/inference-profile.js";
import { applyLlamaEnvMitigation } from "./lib/cli.js";

function usage(exitCode: number): never {
  const message = [
    "Usage:",
    "  npm run bench:expand -- --benchmark <dir> --run <name> --model <id-or-path> [--force]",
    "  npm run bench:expand -- --benchmark <dir> --run <name> --profile <profile.json> --seed <n> [--force]",
    "",
    "Examples:",
    "  npm run bench:expand -- --benchmark finetune/benchmarks/qmd-expansion-scifact-v1 \\",
    "    --run upstream-qmd --model hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf",
    "  npm run bench:expand -- --benchmark finetune/benchmarks/qmd-expansion-scifact-v1 \\",
    "    --run qwen-1.7b-base --model /absolute/path/to/base.gguf",
    "",
    "Profile mode pins the full V2-VH prompt template, grammar, sampling",
    "parameters, and model sha256 from a qmd-inference-profile-vh-v1 file.",
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
    profile: { type: "string" },
    seed: { type: "string" },
    force: { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
  strict: true,
});

if (values.help) usage(0);
if (!values.benchmark || !values.run) usage(1);
if (Boolean(values.model) === Boolean(values.profile)) {
  console.error("provide exactly one of --model or --profile");
  usage(1);
}
if (values.profile && !values.seed) {
  console.error("--profile requires --seed");
  usage(1);
}
if (values.seed && !/^\d+$/.test(values.seed)) {
  console.error("--seed must be a non-negative integer");
  usage(1);
}

applyLlamaEnvMitigation();

const { LlamaCpp } = await import("../../src/llm.js");

if (values.profile) {
  // Profiled V2-VH path: model path/identity comes from the profile itself.
  const profile = loadInferenceProfile(values.profile);
  const seed = Number(values.seed);
  if (!profile.seeds.includes(seed)) {
    console.error(
      `seed ${seed} is not in the profile's frozen seed list [${profile.seeds.join(", ")}]`,
    );
    process.exit(1);
  }
  const llm = new LlamaCpp({
    generateModel: profile.model.path,
    inactivityTimeoutMs: 0,
  });
  try {
    const modelSha = fileSha256(profile.model.path);
    if (modelSha !== profile.model.sha256) {
      console.error(
        `model sha mismatch: ${profile.model.path} hashes to ${modelSha}, profile pins ${profile.model.sha256}`,
      );
      process.exit(1);
    }
    const identity = {
      seed,
      prompt_sha256: profile.prompt.sha256,
      model_sha256: modelSha,
      profile_sha256: inferenceProfileSha256(values.profile),
    };
    const summary = await generateExpansionArtifact({
      benchmarkDir: values.benchmark,
      runName: values.run,
      force: values.force,
      identity,
      generateVh: async query => {
        const { text, meta } = await llm.generateQueryExpansionWithMeta(query, {
          promptTemplate: profile.prompt.template,
          grammar: profile.grammar,
          seed,
          maxTokens: profile.sampling.max_tokens,
          systemPrompt: profile.system_prompt,
          sampling: {
            temperature: profile.sampling.temperature,
            topK: profile.sampling.top_k,
            topP: profile.sampling.top_p,
            presencePenalty: profile.sampling.presence_penalty,
            lastTokens: profile.sampling.repeat_penalty_last_tokens,
          },
        });
        return { raw_output: text, meta };
      },
      onProgress: renderProgress,
    });
    console.log(JSON.stringify({ profile: values.profile, seed, ...summary }, null, 2));
  } finally {
    await llm.dispose();
  }
} else {
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
}
