import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { BenchmarkExpansion } from "../../../src/bench/types.js";

export const OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses";
export const DEEPSEEK_API_BASE_URL = "https://api.deepseek.com";
export const OPENAI_DISTILL_PROMPT_VERSION = "qmd-expansion-teacher-v1";
export const OPENAI_REQUEST_TIMEOUT_MS = 120_000;
export const OPENAI_DEFAULT_MAX_OUTPUT_TOKENS = 4096;
export const OPENAI_LEGACY_MAX_OUTPUT_TOKENS = 1200;

export const OPENAI_REASONING_EFFORTS = [
  "none",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;

export type OpenAiReasoningEffort = typeof OPENAI_REASONING_EFFORTS[number];

export type OpenAiPromptConfig = {
  version: string;
  systemPrompt: string;
  userPromptTemplate: string;
  sha256: string;
  source: "default" | "environment" | "file";
  systemPromptFile: string | null;
  userPromptTemplateFile: string | null;
};

export type OpenAiGenerationErrorCode =
  | "http_error"
  | "timeout"
  | "incomplete_max_tokens"
  | "completed_without_output_text"
  | "refusal"
  | "invalid_json"
  | "schema_error";

export type OpenAiGenerationDiagnostics = {
  attempts: number;
  response_status: string | null;
  incomplete_reason: string | null;
  output_item_types: string[];
  content_block_types: string[];
  output_tokens: number | null;
  reasoning_tokens: number | null;
  http_status?: number;
};

export type OpenAiGeneratedExpansion = {
  raw_output: string;
  parsed_output: BenchmarkExpansion[];
};

const DEFAULT_SYSTEM_PROMPT = [
  "Generate search-query expansions for QMD hybrid retrieval.",
  "Return 1-3 short lexical searches, 1-3 natural-language vector searches,",
  "and optionally one 32-128 token hypothetical relevant passage.",
  "Do not explain your answer and do not include reasoning or chat-template markers.",
].join(" ");
const DEFAULT_USER_PROMPT_TEMPLATE = "Expand this search query: {{query}}";

function promptHash(version: string, systemPrompt: string, userPromptTemplate: string): string {
  return createHash("sha256")
    .update(JSON.stringify({ version, system_prompt: systemPrompt, user_prompt_template: userPromptTemplate }))
    .digest("hex");
}

export function loadOpenAiPromptConfig(
  environment: NodeJS.ProcessEnv = process.env,
): OpenAiPromptConfig {
  const hasVersion = environment.DISTILL_PROMPT_VERSION !== undefined;
  const inlineFields = [
    environment.DISTILL_SYSTEM_PROMPT,
    environment.DISTILL_USER_PROMPT_TEMPLATE,
  ].filter(value => value !== undefined).length;
  const fileFields = [
    environment.DISTILL_SYSTEM_PROMPT_FILE,
    environment.DISTILL_USER_PROMPT_TEMPLATE_FILE,
  ].filter(value => value !== undefined).length;
  if (inlineFields > 0 && fileFields > 0) {
    throw new Error(
      "Prompt text must use either inline variables or file variables, not both",
    );
  }
  const configuredFields = inlineFields > 0 ? inlineFields : fileFields;
  if ((hasVersion || configuredFields > 0) && (!hasVersion || configuredFields !== 2)) {
    throw new Error(
      "DISTILL_PROMPT_VERSION and both inline or file Prompt variables must be set together",
    );
  }
  const source = inlineFields === 2 ? "environment" : fileFields === 2 ? "file" : "default";
  const systemPromptFile = source === "file"
    ? resolve(environment.DISTILL_SYSTEM_PROMPT_FILE!)
    : null;
  const userPromptTemplateFile = source === "file"
    ? resolve(environment.DISTILL_USER_PROMPT_TEMPLATE_FILE!)
    : null;
  const version = environment.DISTILL_PROMPT_VERSION?.trim() || OPENAI_DISTILL_PROMPT_VERSION;
  const systemPrompt = source === "file"
    ? readFileSync(systemPromptFile!, "utf8").trim()
    : environment.DISTILL_SYSTEM_PROMPT?.trim() || DEFAULT_SYSTEM_PROMPT;
  const userPromptTemplate = source === "file"
    ? readFileSync(userPromptTemplateFile!, "utf8").trim()
    : environment.DISTILL_USER_PROMPT_TEMPLATE?.trim() || DEFAULT_USER_PROMPT_TEMPLATE;
  if (!version || !systemPrompt || !userPromptTemplate) {
    throw new Error("Prompt version and Prompt files must not be empty");
  }
  if (!userPromptTemplate.includes("{{query}}")) {
    throw new Error("DISTILL_USER_PROMPT_TEMPLATE must contain {{query}}");
  }
  return {
    version,
    systemPrompt,
    userPromptTemplate,
    sha256: promptHash(version, systemPrompt, userPromptTemplate),
    source,
    systemPromptFile,
    userPromptTemplateFile,
  };
}

export class OpenAiGenerationError extends Error {
  readonly code: OpenAiGenerationErrorCode;
  readonly diagnostics: OpenAiGenerationDiagnostics;
  readonly retryable: boolean;

  constructor(
    code: OpenAiGenerationErrorCode,
    message: string,
    diagnostics: OpenAiGenerationDiagnostics,
    retryable: boolean,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "OpenAiGenerationError";
    this.code = code;
    this.diagnostics = diagnostics;
    this.retryable = retryable;
  }
}

function emptyDiagnostics(): OpenAiGenerationDiagnostics {
  return {
    attempts: 1,
    response_status: null,
    incomplete_reason: null,
    output_item_types: [],
    content_block_types: [],
    output_tokens: null,
    reasoning_tokens: null,
  };
}

function stringField(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberField(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function responseDiagnostics(payload: Record<string, unknown>): OpenAiGenerationDiagnostics {
  const output = Array.isArray(payload.output) ? payload.output : [];
  const outputItemTypes: string[] = [];
  const contentBlockTypes: string[] = [];
  for (const item of output) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const itemType = stringField(record.type);
    if (itemType) outputItemTypes.push(itemType);
    if (!Array.isArray(record.content)) continue;
    for (const content of record.content) {
      if (!content || typeof content !== "object") continue;
      const contentType = stringField((content as Record<string, unknown>).type);
      if (contentType) contentBlockTypes.push(contentType);
    }
  }
  const incompleteDetails = payload.incomplete_details && typeof payload.incomplete_details === "object"
    ? payload.incomplete_details as Record<string, unknown>
    : {};
  const usage = payload.usage && typeof payload.usage === "object"
    ? payload.usage as Record<string, unknown>
    : {};
  const outputDetails = usage.output_tokens_details && typeof usage.output_tokens_details === "object"
    ? usage.output_tokens_details as Record<string, unknown>
    : {};
  return {
    attempts: 1,
    response_status: stringField(payload.status),
    incomplete_reason: stringField(incompleteDetails.reason),
    output_item_types: outputItemTypes,
    content_block_types: contentBlockTypes,
    output_tokens: numberField(usage.output_tokens),
    reasoning_tokens: numberField(outputDetails.reasoning_tokens),
  };
}

function withAttempts(error: OpenAiGenerationError, attempts: number): OpenAiGenerationError {
  error.diagnostics.attempts = attempts;
  return error;
}

function asGenerationError(error: unknown): OpenAiGenerationError {
  if (error instanceof OpenAiGenerationError) return error;
  const timedOut = error instanceof DOMException
    ? error.name === "TimeoutError" || error.name === "AbortError"
    : error instanceof Error && /timed?\s*out|timeout/i.test(error.message);
  return new OpenAiGenerationError(
    timedOut ? "timeout" : "http_error",
    timedOut ? "OpenAI Responses API request timed out" : "OpenAI Responses API request failed",
    emptyDiagnostics(),
    true,
    error instanceof Error ? { cause: error } : undefined,
  );
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

type Fetch = typeof fetch;

const outputSchema = {
  type: "object",
  properties: {
    expansions: {
      type: "array",
      items: {
        type: "object",
        properties: {
          type: { type: "string", enum: ["lex", "vec", "hyde"] },
          query: { type: "string" },
        },
        required: ["type", "query"],
        additionalProperties: false,
      },
    },
  },
  required: ["expansions"],
  additionalProperties: false,
} as const;

function extractOutputText(payload: Record<string, unknown>): string {
  const diagnostics = responseDiagnostics(payload);
  if (payload.status !== "completed") {
    const maxTokens = diagnostics.incomplete_reason === "max_output_tokens"
      || diagnostics.incomplete_reason === "max_tokens";
    throw new OpenAiGenerationError(
      maxTokens ? "incomplete_max_tokens" : "completed_without_output_text",
      maxTokens
        ? "OpenAI response was incomplete because it reached the output token limit"
        : `OpenAI response did not complete (status=${JSON.stringify(payload.status)})`,
      diagnostics,
      true,
    );
  }
  if (!Array.isArray(payload.output)) {
    throw new OpenAiGenerationError(
      "completed_without_output_text",
      "OpenAI response has no output array",
      diagnostics,
      true,
    );
  }

  const text: string[] = [];
  let refused = false;
  for (const item of payload.output) {
    if (!item || typeof item !== "object" || !Array.isArray((item as Record<string, unknown>).content)) {
      continue;
    }
    for (const content of (item as { content: unknown[] }).content) {
      if (!content || typeof content !== "object") continue;
      const block = content as Record<string, unknown>;
      if (block.type === "output_text" && typeof block.text === "string") text.push(block.text);
      if (block.type === "refusal") refused = true;
    }
  }
  if (refused) {
    throw new OpenAiGenerationError(
      "refusal",
      "OpenAI model refused the request",
      diagnostics,
      false,
    );
  }
  if (text.length === 0) {
    throw new OpenAiGenerationError(
      "completed_without_output_text",
      "OpenAI response contains no output_text",
      diagnostics,
      true,
    );
  }
  return text.join("");
}

function parseStructuredExpansion(
  rawOutput: string,
  diagnostics: OpenAiGenerationDiagnostics,
): BenchmarkExpansion[] {
  let value: unknown;
  try {
    value = JSON.parse(rawOutput);
  } catch (error) {
    throw new OpenAiGenerationError(
      "invalid_json",
      "OpenAI structured output is not valid JSON",
      diagnostics,
      true,
      { cause: error },
    );
  }
  let topLevelExpansionsKeys = 0;
  let depth = 0;
  for (let index = 0; index < rawOutput.length; index++) {
    const character = rawOutput[index]!;
    if (character === "{") {
      depth++;
      continue;
    }
    if (character === "}") {
      depth--;
      continue;
    }
    if (character !== '"') continue;
    const start = index;
    for (index++; index < rawOutput.length; index++) {
      if (rawOutput[index] === "\\") {
        index++;
      } else if (rawOutput[index] === '"') {
        break;
      }
    }
    if (depth !== 1 || index >= rawOutput.length) continue;
    let cursor = index + 1;
    while (/\s/.test(rawOutput[cursor] ?? "")) cursor++;
    if (rawOutput[cursor] !== ":") continue;
    if (JSON.parse(rawOutput.slice(start, index + 1)) === "expansions") {
      topLevelExpansionsKeys++;
    }
  }
  if (topLevelExpansionsKeys !== 1) {
    throw new OpenAiGenerationError(
      "schema_error",
      "OpenAI structured output must contain exactly one top-level expansions key",
      diagnostics,
      true,
    );
  }
  if (!value || typeof value !== "object" || !Array.isArray((value as Record<string, unknown>).expansions)) {
    throw new OpenAiGenerationError(
      "schema_error",
      "OpenAI structured output has no expansions array",
      diagnostics,
      true,
    );
  }
  return (value as { expansions: unknown[] }).expansions.map((item, index) => {
    if (!item || typeof item !== "object") {
      throw new OpenAiGenerationError(
        "schema_error",
        `OpenAI expansion ${index} must be an object`,
        diagnostics,
        true,
      );
    }
    const expansion = item as Record<string, unknown>;
    if (
      (expansion.type !== "lex" && expansion.type !== "vec" && expansion.type !== "hyde")
      || typeof expansion.query !== "string"
      || !expansion.query.trim()
    ) {
      throw new OpenAiGenerationError(
        "schema_error",
        `OpenAI expansion ${index} has an invalid type or query`,
        diagnostics,
        true,
      );
    }
    return [expansion.type, expansion.query] as BenchmarkExpansion;
  });
}

export async function generateOpenAiExpansion(options: {
  apiKey: string;
  endpoint?: string;
  model: string;
  query: string;
  reasoningEffort: OpenAiReasoningEffort;
  maxOutputTokens?: number;
  prompt?: OpenAiPromptConfig;
  maxAttempts?: number;
  retryDelayMs?: number;
  fetchImpl?: Fetch;
}): Promise<OpenAiGeneratedExpansion> {
  const maxAttempts = options.maxAttempts ?? 1;
  const maxOutputTokens = options.maxOutputTokens ?? OPENAI_DEFAULT_MAX_OUTPUT_TOKENS;
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts <= 0) {
    throw new Error("maxAttempts must be a positive integer");
  }
  if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens <= 0) {
    throw new Error("maxOutputTokens must be a positive integer");
  }
  for (let attempt = 1; ; attempt++) {
    try {
      return await generateOpenAiExpansionOnce(options);
    } catch (error) {
      const generationError = withAttempts(asGenerationError(error), attempt);
      if (attempt >= maxAttempts || !generationError.retryable) throw generationError;
      await sleep((options.retryDelayMs ?? 1_000) * attempt);
    }
  }
}

async function generateOpenAiExpansionOnce(options: {
  apiKey: string;
  endpoint?: string;
  model: string;
  query: string;
  reasoningEffort: OpenAiReasoningEffort;
  maxOutputTokens?: number;
  prompt?: OpenAiPromptConfig;
  fetchImpl?: Fetch;
}): Promise<OpenAiGeneratedExpansion> {
  const fetchImpl = options.fetchImpl ?? fetch;
  const prompt = options.prompt ?? loadOpenAiPromptConfig();
  const maxOutputTokens = options.maxOutputTokens ?? OPENAI_DEFAULT_MAX_OUTPUT_TOKENS;
  let response: Response;
  try {
    response = await fetchImpl(options.endpoint ?? OPENAI_RESPONSES_ENDPOINT, {
      method: "POST",
      signal: AbortSignal.timeout(OPENAI_REQUEST_TIMEOUT_MS),
      headers: {
        Authorization: `Bearer ${options.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: options.model,
        store: false,
        reasoning: { effort: options.reasoningEffort },
        max_output_tokens: maxOutputTokens,
        input: [
          { role: "system", content: prompt.systemPrompt },
          {
            role: "user",
            content: prompt.userPromptTemplate.replaceAll("{{query}}", options.query),
          },
        ],
        text: {
          format: {
            type: "json_schema",
            name: "qmd_query_expansion",
            strict: true,
            schema: outputSchema,
          },
        },
      }),
    });
  } catch (error) {
    throw asGenerationError(error);
  }
  if (!response.ok) {
    const diagnostics = emptyDiagnostics();
    diagnostics.http_status = response.status;
    throw new OpenAiGenerationError(
      "http_error",
      `OpenAI Responses API returned HTTP ${response.status}`,
      diagnostics,
      response.status === 408 || response.status === 409 || response.status === 429 || response.status >= 500,
    );
  }
  let payload: Record<string, unknown>;
  try {
    payload = await response.json() as Record<string, unknown>;
  } catch (error) {
    throw new OpenAiGenerationError(
      "invalid_json",
      "OpenAI Responses API returned invalid JSON",
      emptyDiagnostics(),
      true,
      { cause: error },
    );
  }
  const rawOutput = extractOutputText(payload);
  return {
    raw_output: rawOutput,
    parsed_output: parseStructuredExpansion(rawOutput, responseDiagnostics(payload)),
  };
}

function chatCompletionDiagnostics(payload: Record<string, unknown>): OpenAiGenerationDiagnostics {
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const first = choices[0] && typeof choices[0] === "object"
    ? choices[0] as Record<string, unknown>
    : {};
  const message = first.message && typeof first.message === "object"
    ? first.message as Record<string, unknown>
    : {};
  const usage = payload.usage && typeof payload.usage === "object"
    ? payload.usage as Record<string, unknown>
    : {};
  const completionDetails = usage.completion_tokens_details
    && typeof usage.completion_tokens_details === "object"
    ? usage.completion_tokens_details as Record<string, unknown>
    : {};
  const finishReason = stringField(first.finish_reason);
  return {
    attempts: 1,
    response_status: choices.length > 0 ? "completed" : null,
    incomplete_reason: finishReason === "length" ? "max_tokens" : null,
    output_item_types: choices.length > 0 ? ["choice"] : [],
    content_block_types: typeof message.content === "string" && message.content
      ? ["message.content"]
      : [],
    output_tokens: numberField(usage.completion_tokens),
    reasoning_tokens: numberField(completionDetails.reasoning_tokens),
  };
}

function chatCompletionOutput(payload: Record<string, unknown>): {
  rawOutput: string;
  diagnostics: OpenAiGenerationDiagnostics;
} {
  const diagnostics = chatCompletionDiagnostics(payload);
  const choices = Array.isArray(payload.choices) ? payload.choices : [];
  const first = choices[0] && typeof choices[0] === "object"
    ? choices[0] as Record<string, unknown>
    : {};
  const finishReason = stringField(first.finish_reason);
  if (finishReason === "length") {
    throw new OpenAiGenerationError(
      "incomplete_max_tokens",
      "Chat completion reached the output token limit",
      diagnostics,
      true,
    );
  }
  if (finishReason === "content_filter") {
    throw new OpenAiGenerationError(
      "refusal",
      "Chat completion was blocked by the content filter",
      diagnostics,
      false,
    );
  }
  const message = first.message && typeof first.message === "object"
    ? first.message as Record<string, unknown>
    : {};
  if (typeof message.content !== "string" || !message.content.trim()) {
    throw new OpenAiGenerationError(
      "completed_without_output_text",
      "Chat completion contains no message content",
      diagnostics,
      true,
    );
  }
  return { rawOutput: message.content, diagnostics };
}

export function openAiChatCompletionsEndpoint(baseUrl: string): string {
  return `${baseUrl.replace(/\/+$/, "")}/chat/completions`;
}

export async function generateOpenAiChatExpansion(options: {
  apiKey: string;
  baseUrl?: string;
  model: string;
  query: string;
  maxOutputTokens?: number;
  prompt?: OpenAiPromptConfig;
  thinkingMode?: "enabled" | "disabled";
  reasoningEffort?: "high" | "max";
  maxAttempts?: number;
  retryDelayMs?: number;
  fetchImpl?: Fetch;
}): Promise<OpenAiGeneratedExpansion> {
  const maxAttempts = options.maxAttempts ?? 1;
  const maxOutputTokens = options.maxOutputTokens ?? OPENAI_DEFAULT_MAX_OUTPUT_TOKENS;
  if (!Number.isSafeInteger(maxAttempts) || maxAttempts <= 0) {
    throw new Error("maxAttempts must be a positive integer");
  }
  if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens <= 0) {
    throw new Error("maxOutputTokens must be a positive integer");
  }
  for (let attempt = 1; ; attempt++) {
    try {
      const fetchImpl = options.fetchImpl ?? fetch;
      const prompt = options.prompt ?? loadOpenAiPromptConfig();
      const thinkingMode = options.thinkingMode ?? "disabled";
      const body: Record<string, unknown> = {
        model: options.model,
        messages: [
          { role: "system", content: prompt.systemPrompt },
          {
            role: "user",
            content: prompt.userPromptTemplate.replaceAll("{{query}}", options.query),
          },
        ],
        max_tokens: maxOutputTokens,
        response_format: { type: "json_object" },
        thinking: { type: thinkingMode },
      };
      if (thinkingMode === "enabled") body.reasoning_effort = options.reasoningEffort ?? "high";
      let response: Response;
      try {
        response = await fetchImpl(
          openAiChatCompletionsEndpoint(options.baseUrl ?? DEEPSEEK_API_BASE_URL),
          {
            method: "POST",
            signal: AbortSignal.timeout(OPENAI_REQUEST_TIMEOUT_MS),
            headers: {
              Authorization: `Bearer ${options.apiKey}`,
              "Content-Type": "application/json",
            },
            body: JSON.stringify(body),
          },
        );
      } catch (error) {
        throw asGenerationError(error);
      }
      if (!response.ok) {
        const diagnostics = emptyDiagnostics();
        diagnostics.http_status = response.status;
        throw new OpenAiGenerationError(
          "http_error",
          `Chat Completions API returned HTTP ${response.status}`,
          diagnostics,
          response.status === 408
            || response.status === 409
            || response.status === 429
            || response.status >= 500,
        );
      }
      let payload: Record<string, unknown>;
      try {
        payload = await response.json() as Record<string, unknown>;
      } catch (error) {
        throw new OpenAiGenerationError(
          "invalid_json",
          "Chat Completions API returned invalid JSON",
          emptyDiagnostics(),
          true,
          { cause: error },
        );
      }
      const { rawOutput, diagnostics } = chatCompletionOutput(payload);
      return {
        raw_output: rawOutput,
        parsed_output: parseStructuredExpansion(rawOutput, diagnostics),
      };
    } catch (error) {
      const generationError = withAttempts(asGenerationError(error), attempt);
      if (attempt >= maxAttempts || !generationError.retryable) throw generationError;
      await sleep((options.retryDelayMs ?? 1_000) * attempt);
    }
  }
}
