/**
 * V2-VH serving inference profile: a self-contained JSON contract pinning the
 * prompt template, grammar, sampling parameters, and model identity for the
 * serving-baseline evaluation. The profile does not depend on the training
 * JSONL at runtime — the template is embedded verbatim and verified by hash.
 *
 * Schema: "qmd-inference-profile-vh-v1"
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

export const INFERENCE_PROFILE_SCHEMA = "qmd-inference-profile-vh-v1";

export interface InferenceProfileVh {
  schema: typeof INFERENCE_PROFILE_SCHEMA;
  /** Model identity: path + sha256 of the exact GGUF artifact to load. */
  model: {
    path: string;
    sha256: string;
    quantization: "Q4_K_M" | "Q8_0" | "F16";
  };
  /**
   * System message for the chat session. Defaults to "" (no system turn),
   * matching the SFT training envelope; node-llama-cpp would otherwise inject
   * its own default system prompt.
   */
  system_prompt: string;
  /** Full prompt template; {query} is the placeholder. */
  prompt: {
    version: string;
    template: string;
    sha256: string;
  };
  /** GBNF grammar constraining the response (1 hyde, then 1-2 vec). */
  grammar: string;
  sampling: {
    temperature: number;
    top_k: number;
    top_p: number;
    presence_penalty: number;
    repeat_penalty_last_tokens: number;
    max_tokens: number;
  };
  seeds: number[];
  fallback_policy: "empty_expansions_raw_query";
  schema_version: typeof EXPANSION_VH_SCHEMA;
}

export const EXPANSION_VH_SCHEMA = "qmd-expansion-vh-serving-v1";

function sha256Text(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function expectString(
  value: unknown,
  location: string,
): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`profile.${location}: expected a non-empty string`);
  }
  return value;
}

function expectNumber(
  value: unknown,
  location: string,
): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`profile.${location}: expected a finite number`);
  }
  return value;
}

/** Load and fully validate a V2-VH inference profile JSON file. */
export function loadInferenceProfile(path: string): InferenceProfileVh {
  const bytes = readFileSync(path);
  let raw: unknown;
  try {
    raw = JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(
      `profile is not valid JSON: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new Error("profile: expected a JSON object");
  }
  const item = raw as Record<string, unknown>;

  if (item.schema !== INFERENCE_PROFILE_SCHEMA) {
    throw new Error(
      `profile.schema: expected "${INFERENCE_PROFILE_SCHEMA}", got ${JSON.stringify(item.schema)}`,
    );
  }

  // Model identity
  if (typeof item.model !== "object" || item.model === null) {
    throw new Error("profile.model: expected an object");
  }
  const model = item.model as Record<string, unknown>;
  const modelPath = expectString(model.path, "model.path");
  const modelSha = expectString(model.sha256, "model.sha256");
  if (!/^[0-9a-f]{64}$/.test(modelSha)) {
    throw new Error("profile.model.sha256: expected a lowercase sha256 hex string");
  }
  if (model.quantization !== "Q4_K_M" && model.quantization !== "Q8_0" && model.quantization !== "F16") {
    throw new Error(`profile.model.quantization: expected "Q4_K_M", "Q8_0" or "F16", got ${JSON.stringify(model.quantization)}`);
  }
  const quantization = model.quantization;

  const systemPrompt = item.system_prompt ?? "";
  if (typeof systemPrompt !== "string") {
    throw new Error("profile.system_prompt: expected a string");
  }

  // Prompt template + hash self-verification
  if (typeof item.prompt !== "object" || item.prompt === null) {
    throw new Error("profile.prompt: expected an object");
  }
  const prompt = item.prompt as Record<string, unknown>;
  const template = expectString(prompt.template, "prompt.template");
  if (!template.includes("{query}")) {
    throw new Error("profile.prompt.template: missing {query} placeholder");
  }
  const promptSha = expectString(prompt.sha256, "prompt.sha256");
  const actualSha = sha256Text(template);
  if (actualSha !== promptSha) {
    throw new Error(
      `profile.prompt.sha256 mismatch: embedded template hashes to ${actualSha}, profile pins ${promptSha}`,
    );
  }
  expectString(prompt.version, "prompt.version");

  // Grammar
  const grammar = expectString(item.grammar, "grammar");
  if (!grammar.includes("hyde") || !grammar.includes("vec")) {
    throw new Error('profile.grammar: missing "hyde"/"vec" rules');
  }
  if (/"lex\b|lex\s*::=/.test(grammar)) {
    throw new Error("profile.grammar: VH contract forbids lex lines");
  }

  // Sampling
  if (typeof item.sampling !== "object" || item.sampling === null) {
    throw new Error("profile.sampling: expected an object");
  }
  const sampling = item.sampling as Record<string, unknown>;
  const temperature = expectNumber(sampling.temperature, "sampling.temperature");
  const topK = expectNumber(sampling.top_k, "sampling.top_k");
  const topP = expectNumber(sampling.top_p, "sampling.top_p");
  const presencePenalty = expectNumber(sampling.presence_penalty, "sampling.presence_penalty");
  const lastTokens = expectNumber(sampling.repeat_penalty_last_tokens, "sampling.repeat_penalty_last_tokens");
  const maxTokens = expectNumber(sampling.max_tokens, "sampling.max_tokens");
  for (const [name, value] of [
    ["top_k", topK],
    ["max_tokens", maxTokens],
  ] as const) {
    if (!Number.isInteger(value) || value <= 0) {
      throw new Error(`profile.sampling.${name}: expected a positive integer`);
    }
  }
  // 0 disables the repeat/presence penalty window entirely.
  if (!Number.isInteger(lastTokens) || lastTokens < 0) {
    throw new Error("profile.sampling.repeat_penalty_last_tokens: expected a non-negative integer");
  }
  if (temperature < 0 || temperature > 2) {
    throw new Error("profile.sampling.temperature: expected a value in [0, 2]");
  }

  // Seeds
  if (!Array.isArray(item.seeds) || item.seeds.length === 0) {
    throw new Error("profile.seeds: expected a non-empty array");
  }
  const seeds = item.seeds.map((seed, index) => {
    if (typeof seed !== "number" || !Number.isInteger(seed) || seed < 0) {
      throw new Error(`profile.seeds[${index}]: expected a non-negative integer`);
    }
    return seed;
  });
  if (new Set(seeds).size !== seeds.length) {
    throw new Error("profile.seeds: duplicate seeds");
  }

  // Fallback policy + expansion schema version
  if (item.fallback_policy !== "empty_expansions_raw_query") {
    throw new Error(
      `profile.fallback_policy: expected "empty_expansions_raw_query", got ${JSON.stringify(item.fallback_policy)}`,
    );
  }
  if (item.expansion_schema !== EXPANSION_VH_SCHEMA) {
    throw new Error(
      `profile.expansion_schema: expected "${EXPANSION_VH_SCHEMA}", got ${JSON.stringify(item.expansion_schema)}`,
    );
  }

  return {
    schema: INFERENCE_PROFILE_SCHEMA,
    model: { path: modelPath, sha256: modelSha, quantization },
    system_prompt: systemPrompt,
    prompt: {
      version: prompt.version as string,
      template,
      sha256: promptSha,
    },
    grammar,
    sampling: {
      temperature,
      top_k: topK,
      top_p: topP,
      presence_penalty: presencePenalty,
      repeat_penalty_last_tokens: lastTokens,
      max_tokens: maxTokens,
    },
    seeds,
    fallback_policy: "empty_expansions_raw_query",
    schema_version: EXPANSION_VH_SCHEMA,
  };
}

/** SHA-256 of the raw profile file, for run manifests and resume identity. */
export function inferenceProfileSha256(path: string): string {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}
