#!/usr/bin/env node
/**
 * Generate expansions with the official tobil/qmd-query-expansion model in its
 * best-case setup: exact training prompt (Qwen3 chat text, no system turn), the
 * empty think block the model emits on its own pre-filled, then a lex/vec/hyde
 * line grammar with an optional final newline, greedy, fixed seed. (qmd's default
 * grammar forces a trailing newline, so the model leaks "</think>" into its last line.)
 * Writes the legacy expansion schema (lex lines kept) plus a sidecar
 * <run>.meta.jsonl with stop_reason / tokens / latency per query.
 *
 *   node_modules/.bin/tsx finetune/experiments/official-model-eval/expand-official.ts \
 *     --benchmark <dir> --run <name> --model <gguf path> [--force]
 */
import { appendFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { platform } from "node:os";

const { values } = parseArgs({
  options: {
    benchmark: { type: "string" },
    run: { type: "string" },
    model: { type: "string" },
    force: { type: "boolean", default: false },
  },
  strict: true,
});
if (!values.benchmark || !values.run || !values.model) {
  console.error("usage: --benchmark <dir> --run <name> --model <gguf> [--force]");
  process.exit(1);
}
if (platform() === "darwin" && !process.env.GGML_METAL_NO_RESIDENCY) {
  process.env.GGML_METAL_NO_RESIDENCY = "1";
}
const { getLlama, LlamaCompletion } = await import("node-llama-cpp");
const { generateExpansionArtifact } = await import("../../../src/bench/expansions.js");

const SEED = 42;
const prompt = (query: string) =>
  `<|im_start|>user\n/no_think Expand this search query: ${query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
// qmd's default line grammar (src/llm.ts), except the final newline is optional:
// training targets end without one, and forcing it makes the model emit "</think>"
// as content before it can stop.
const GRAMMAR = `
root ::= line ("\\n" line)* "\\n"?
line ::= type ": " content
type ::= "lex" | "vec" | "hyde"
content ::= [^\\n]+
`;
const metaPath = join(resolve(values.benchmark), "expansions", `${values.run}.meta.jsonl`);

const llama = await getLlama();
const model = await llama.loadModel({ modelPath: values.model });
const grammar = await llama.createGrammar({ grammar: GRAMMAR });
try {
  const summary = await generateExpansionArtifact({
    benchmarkDir: values.benchmark,
    runName: values.run,
    force: values.force,
    generateRaw: async query => {
      const context = await model.createContext({ contextSize: 2048 });
      try {
        const sequence = context.getSequence();
        const completion = new LlamaCompletion({ contextSequence: sequence });
        const start = performance.now();
        const before = sequence.tokenMeter.usedOutputTokens;
        const { response, metadata } = await completion.generateCompletionWithMeta(
          model.tokenize(prompt(query), true),
          { grammar, temperature: 0, seed: SEED, maxTokens: 600 },
        );
        appendFileSync(metaPath, `${JSON.stringify({
          query,
          stop_reason: metadata.stopReason,
          generated_tokens: sequence.tokenMeter.usedOutputTokens - before,
          elapsed_ms: Math.round(performance.now() - start),
        })}\n`);
        return response;
      } finally {
        await context.dispose();
      }
    },
    onProgress: p => {
      if (p.completed % 50 === 0 || p.completed === p.total) {
        console.error(`${p.completed}/${p.total} elapsed=${Math.round(p.elapsed_ms / 1000)}s format=${p.format_errors} gen=${p.generation_errors}`);
      }
    },
  });
  console.log(JSON.stringify({ model: values.model, seed: SEED, prefill: "<think>\\n\\n</think>\\n\\n", ...summary }, null, 2));
} finally {
  await model.dispose();
}
