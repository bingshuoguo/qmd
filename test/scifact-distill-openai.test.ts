import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import {
  DEEPSEEK_API_BASE_URL,
  OPENAI_DEFAULT_MAX_OUTPUT_TOKENS,
  OpenAiGenerationError,
  OPENAI_RESPONSES_ENDPOINT,
  generateOpenAiExpansion,
  generateOpenAiChatExpansion,
  loadOpenAiPromptConfig,
} from "../finetune/benchmarks/lib/openai-teacher.js";

function completedResponse(output: unknown): Response {
  return new Response(JSON.stringify({
    id: "resp_test",
    status: "completed",
    output: [{
      type: "message",
      content: [{ type: "output_text", text: JSON.stringify(output) }],
    }],
  }), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("OpenAI SciFact distillation teacher", () => {
  test("uses Structured Outputs and returns the existing expansion tuple schema", async () => {
    const fetchMock = vi.fn(async () => completedResponse({
      expansions: [
        { type: "lex", query: "vitamin d randomized trial" },
        { type: "vec", query: "evidence about vitamin d in randomized trials" },
      ],
    }));

    const generated = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "Does vitamin D help?",
      reasoningEffort: "low",
      fetchImpl: fetchMock,
    });

    expect(generated.parsed_output).toEqual([
      ["lex", "vitamin d randomized trial"],
      ["vec", "evidence about vitamin d in randomized trials"],
    ]);
    expect(JSON.parse(generated.raw_output)).toEqual({
      expansions: [
        { type: "lex", query: "vitamin d randomized trial" },
        { type: "vec", query: "evidence about vitamin d in randomized trials" },
      ],
    });
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe(OPENAI_RESPONSES_ENDPOINT);
    expect((init!.headers as Record<string, string>).Authorization).toBe("Bearer secret-test-key");
    const body = JSON.parse(init!.body as string);
    expect(body).toMatchObject({
      model: "gpt-test",
      store: false,
      reasoning: { effort: "low" },
      max_output_tokens: OPENAI_DEFAULT_MAX_OUTPUT_TOKENS,
      text: { format: { type: "json_schema", strict: true } },
    });
  });

  test("uses configurable prompt text, version, and output budget", async () => {
    const prompt = loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v2",
      DISTILL_SYSTEM_PROMPT: "Preserve biomedical semantics.",
      DISTILL_USER_PROMPT_TEMPLATE: "Query={{query}}",
    });
    const fetchMock = vi.fn(async () => completedResponse({ expansions: [] }));

    await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "alpha",
      reasoningEffort: "low",
      maxOutputTokens: 4096,
      prompt,
      fetchImpl: fetchMock,
    });

    expect(prompt).toMatchObject({
      version: "qmd-expansion-teacher-v2",
      source: "environment",
    });
    expect(prompt.sha256).toMatch(/^[a-f0-9]{64}$/);
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body.max_output_tokens).toBe(4096);
    expect(body.input).toEqual([
      { role: "system", content: "Preserve biomedical semantics." },
      { role: "user", content: "Query=alpha" },
    ]);
  });

  test("rejects a user prompt template that cannot interpolate the query", () => {
    expect(() => loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "test",
      DISTILL_SYSTEM_PROMPT: "test",
      DISTILL_USER_PROMPT_TEMPLATE: "No placeholder",
    })).toThrow("must contain {{query}}");
  });

  test("rejects partially configured prompt environments", () => {
    expect(() => loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v2",
    })).toThrow("must be set together");
  });

  test("loads complete prompts from files without dotenv quote truncation", () => {
    const directory = mkdtempSync(join(tmpdir(), "qmd-prompt-"));
    const systemPath = join(directory, "system.txt");
    const userPath = join(directory, "user.txt");
    const systemPrompt = [
      "Return JSON only.",
      "Valid shape: {\"expansions\":[{\"type\":\"lex\",\"query\":\"...\"}]}",
      "Preserve negation and numbers.",
    ].join("\n");
    writeFileSync(systemPath, `${systemPrompt}\n`, "utf8");
    writeFileSync(userPath, "Input: {{query}}\n", "utf8");
    try {
      const prompt = loadOpenAiPromptConfig({
        DISTILL_PROMPT_VERSION: "file-prompt-v1",
        DISTILL_SYSTEM_PROMPT_FILE: systemPath,
        DISTILL_USER_PROMPT_TEMPLATE_FILE: userPath,
      });
      expect(prompt).toMatchObject({
        version: "file-prompt-v1",
        source: "file",
        systemPrompt,
        userPromptTemplate: "Input: {{query}}",
        systemPromptFile: systemPath,
        userPromptTemplateFile: userPath,
      });
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });

  test("loads the checked-in OpenAI V2 RAG prompt configuration", () => {
    const systemPath = join(
      import.meta.dirname,
      "../finetune/benchmarks/distill/prompts/openai-v2-system.txt",
    );
    const userPath = join(
      import.meta.dirname,
      "../finetune/benchmarks/distill/prompts/openai-v2-user.txt",
    );
    const prompt = loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v2-openai-rag-semantic-safe",
      DISTILL_SYSTEM_PROMPT_FILE: systemPath,
      DISTILL_USER_PROMPT_TEMPLATE_FILE: userPath,
    });

    expect(prompt).toMatchObject({
      version: "qmd-expansion-teacher-v2-openai-rag-semantic-safe",
      source: "file",
      systemPrompt: readFileSync(systemPath, "utf8").trim(),
      userPromptTemplate: readFileSync(userPath, "utf8").trim(),
    });
    expect(prompt.systemPrompt).toContain("hybrid RAG retrieval system");
    expect(prompt.systemPrompt).toContain("unverified claim");
    expect(prompt.systemPrompt).toContain("Preserve entities, subject-object roles, negation");
    expect(prompt.userPromptTemplate).toContain("<query>{{query}}</query>");
    expect(prompt.sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  test("loads the checked-in DeepSeek V2 prompt configuration", () => {
    const systemPath = join(
      import.meta.dirname,
      "../finetune/benchmarks/distill/prompts/deepseek-v2-system.txt",
    );
    const userPath = join(
      import.meta.dirname,
      "../finetune/benchmarks/distill/prompts/deepseek-v2-user.txt",
    );
    const prompt = loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v2-deepseek-general-purpose",
      DISTILL_SYSTEM_PROMPT_FILE: systemPath,
      DISTILL_USER_PROMPT_TEMPLATE_FILE: userPath,
    });

    expect(prompt).toMatchObject({
      version: "qmd-expansion-teacher-v2-deepseek-general-purpose",
      source: "file",
      systemPrompt: readFileSync(systemPath, "utf8").trim(),
      userPromptTemplate: readFileSync(userPath, "utf8").trim(),
    });
    expect(prompt.systemPrompt).toContain("1-3 distinct lex expansions");
    expect(prompt.systemPrompt).toContain("32-128 tokens");
    expect(prompt.systemPrompt).toContain("Never convert semantic negation");
    expect(prompt.userPromptTemplate).toContain("<query>{{query}}</query>");
    expect(prompt.sha256).toMatch(/^[a-f0-9]{64}$/);
  });

  test("loads the fixed-count SciFact V3 prompt configuration", () => {
    const systemPath = join(
      import.meta.dirname,
      "../finetune/benchmarks/distill/prompts/deepseek-v3-scifact-system.txt",
    );
    const userPath = join(
      import.meta.dirname,
      "../finetune/benchmarks/distill/prompts/deepseek-v3-scifact-user.txt",
    );
    const prompt = loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v3-deepseek-scifact-unverified-claim",
      DISTILL_SYSTEM_PROMPT_FILE: systemPath,
      DISTILL_USER_PROMPT_TEMPLATE_FILE: userPath,
    });

    expect(prompt.systemPrompt).toContain("Every input is an unverified scientific claim");
    expect(prompt.systemPrompt).toContain("exactly 3 lex + 3 vec + 1 hyde");
    expect(prompt.systemPrompt).toContain("32-128 tokens");
    expect(prompt.userPromptTemplate).toContain("<claim>{{query}}</claim>");
  });

  test("rejects mixed inline and file Prompt configuration", () => {
    expect(() => loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "mixed-v1",
      DISTILL_SYSTEM_PROMPT: "inline",
      DISTILL_USER_PROMPT_TEMPLATE: "{{query}}",
      DISTILL_SYSTEM_PROMPT_FILE: "system.txt",
      DISTILL_USER_PROMPT_TEMPLATE_FILE: "user.txt",
    })).toThrow("not both");
  });

  test("uses a configured OpenAI-compatible Responses endpoint", async () => {
    const endpoint = "https://gateway.example/v1/responses";
    const fetchMock = vi.fn(async () => completedResponse({
      expansions: [
        { type: "lex", query: "alpha evidence" },
        { type: "vec", query: "evidence about alpha" },
      ],
    }));

    await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      endpoint,
      model: "gateway-model",
      query: "alpha",
      reasoningEffort: "low",
      fetchImpl: fetchMock,
    });

    expect(fetchMock.mock.calls[0]![0]).toBe(endpoint);
  });

  test("reports refusals as generation errors instead of training targets", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      status: "completed",
      output: [{ content: [{ type: "refusal", refusal: "cannot comply" }] }],
    }), { status: 200 }));

    const error = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "query",
      reasoningEffort: "low",
      fetchImpl: fetchMock,
    }).catch(value => value as OpenAiGenerationError);

    expect(error.code).toBe("refusal");
    expect(error.diagnostics.content_block_types).toEqual(["refusal"]);
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  test("retries a completed response without output text", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "completed", output: [] }), {
        status: 200,
      }))
      .mockResolvedValueOnce(completedResponse({
        expansions: [
          { type: "lex", query: "alpha evidence" },
          { type: "vec", query: "evidence about alpha" },
        ],
      }));

    const generated = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "alpha",
      reasoningEffort: "low",
      maxAttempts: 2,
      retryDelayMs: 0,
      fetchImpl: fetchMock,
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(generated.parsed_output[0]).toEqual(["lex", "alpha evidence"]);
  });

  test("does not include the API key in HTTP failure errors", async () => {
    const fetchMock = vi.fn(async () => new Response("rate limited", { status: 429 }));

    const error = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "query",
      reasoningEffort: "low",
      fetchImpl: fetchMock,
    }).catch(value => value as Error);

    expect(error.message).toContain("HTTP 429");
    expect(error.message).not.toContain("secret-test-key");
    expect(error.message).not.toContain("rate limited");
  });

  test("classifies completed reasoning-only output and exposes safe diagnostics", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      status: "completed",
      output: [{ type: "reasoning", content: [] }],
      usage: { output_tokens: 1200, output_tokens_details: { reasoning_tokens: 1200 } },
    }), { status: 200 }));

    const error = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "query",
      reasoningEffort: "low",
      maxAttempts: 3,
      retryDelayMs: 0,
      fetchImpl: fetchMock,
    }).catch(value => value as OpenAiGenerationError);

    expect(error.code).toBe("completed_without_output_text");
    expect(error.diagnostics).toEqual({
      attempts: 3,
      response_status: "completed",
      incomplete_reason: null,
      output_item_types: ["reasoning"],
      content_block_types: [],
      output_tokens: 1200,
      reasoning_tokens: 1200,
    });
    expect(JSON.stringify(error.diagnostics)).not.toContain("secret-test-key");
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  test("classifies incomplete max-token responses", async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" },
      output: [{ type: "reasoning", content: [] }],
    }), { status: 200 }));

    const error = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "query",
      reasoningEffort: "low",
      fetchImpl: fetchMock,
    }).catch(value => value as OpenAiGenerationError);

    expect(error.code).toBe("incomplete_max_tokens");
    expect(error.diagnostics.incomplete_reason).toBe("max_output_tokens");
  });

  test("classifies request timeouts without exposing the underlying request", async () => {
    const fetchMock = vi.fn(async () => {
      throw new DOMException("request aborted", "TimeoutError");
    });

    const error = await generateOpenAiExpansion({
      apiKey: "secret-test-key",
      model: "gpt-test",
      query: "query",
      reasoningEffort: "low",
      fetchImpl: fetchMock,
    }).catch(value => value as OpenAiGenerationError);

    expect(error.code).toBe("timeout");
    expect(error.message).toBe("OpenAI Responses API request timed out");
    expect(JSON.stringify(error.diagnostics)).not.toContain("secret-test-key");
  });

  test("classifies invalid JSON and schema failures", async () => {
    const invalidJson = vi.fn(async () => new Response(JSON.stringify({
      status: "completed",
      output: [{ type: "message", content: [{ type: "output_text", text: "{" }] }],
    }), { status: 200 }));
    const invalidSchema = vi.fn(async () => completedResponse({ wrong: [] }));

    const jsonError = await generateOpenAiExpansion({
      apiKey: "key",
      model: "model",
      query: "query",
      reasoningEffort: "low",
      fetchImpl: invalidJson,
    }).catch(value => value as OpenAiGenerationError);
    const schemaError = await generateOpenAiExpansion({
      apiKey: "key",
      model: "model",
      query: "query",
      reasoningEffort: "low",
      fetchImpl: invalidSchema,
    }).catch(value => value as OpenAiGenerationError);

    expect(jsonError.code).toBe("invalid_json");
    expect(schemaError.code).toBe("schema_error");
  });

  test("rejects duplicate top-level expansions keys before accepting parsed JSON", async () => {
    const duplicate = vi.fn(async () => new Response(JSON.stringify({
      status: "completed",
      output: [{
        type: "message",
        content: [{
          type: "output_text",
          text: '{"expansions":[],"expansions":[{"type":"lex","query":"alpha"}]}',
        }],
      }],
    }), { status: 200 }));

    const error = await generateOpenAiExpansion({
      apiKey: "key",
      model: "model",
      query: "query",
      reasoningEffort: "low",
      maxAttempts: 1,
      fetchImpl: duplicate,
    }).catch(value => value as OpenAiGenerationError);

    expect(error.code).toBe("schema_error");
    expect(error.message).toContain("exactly one top-level expansions key");
  });

  test("retries 429 and 5xx but stops immediately on non-retryable 4xx", async () => {
    const retrying = vi
      .fn()
      .mockResolvedValueOnce(new Response("", { status: 429 }))
      .mockResolvedValueOnce(new Response("", { status: 503 }))
      .mockResolvedValueOnce(completedResponse({ expansions: [] }));
    await generateOpenAiExpansion({
      apiKey: "key",
      model: "model",
      query: "query",
      reasoningEffort: "low",
      maxAttempts: 3,
      retryDelayMs: 0,
      fetchImpl: retrying,
    });
    expect(retrying).toHaveBeenCalledTimes(3);

    const notRetrying = vi.fn(async () => new Response("", { status: 400 }));
    const error = await generateOpenAiExpansion({
      apiKey: "key",
      model: "model",
      query: "query",
      reasoningEffort: "low",
      maxAttempts: 3,
      retryDelayMs: 0,
      fetchImpl: notRetrying,
    }).catch(value => value as OpenAiGenerationError);
    expect(error.code).toBe("http_error");
    expect(error.diagnostics.attempts).toBe(1);
    expect(notRetrying).toHaveBeenCalledOnce();
  });
});

describe("OpenAI-compatible Chat Completions distillation teacher", () => {
  test("uses the DeepSeek endpoint, JSON mode, configured prompt, and disabled thinking", async () => {
    const prompt = loadOpenAiPromptConfig({
      DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v1-deepseek",
      DISTILL_SYSTEM_PROMPT: "Return JSON expansions.",
      DISTILL_USER_PROMPT_TEMPLATE: "Expand {{query}}",
    });
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({
      choices: [{
        finish_reason: "stop",
        message: {
          content: JSON.stringify({
            expansions: [{ type: "lex", query: "alpha evidence" }],
          }),
        },
      }],
      usage: { completion_tokens: 20 },
    }), { status: 200 }));

    const generated = await generateOpenAiChatExpansion({
      apiKey: "secret-test-key",
      model: "deepseek-v4-flash",
      query: "alpha",
      maxOutputTokens: 4096,
      prompt,
      thinkingMode: "disabled",
      fetchImpl: fetchMock,
    });

    expect(generated.parsed_output).toEqual([["lex", "alpha evidence"]]);
    expect(fetchMock.mock.calls[0]![0]).toBe(`${DEEPSEEK_API_BASE_URL}/chat/completions`);
    const body = JSON.parse(fetchMock.mock.calls[0]![1]!.body as string);
    expect(body).toMatchObject({
      model: "deepseek-v4-flash",
      max_tokens: 4096,
      response_format: { type: "json_object" },
      thinking: { type: "disabled" },
      messages: [
        { role: "system", content: "Return JSON expansions." },
        { role: "user", content: "Expand alpha" },
      ],
    });
    expect(body).not.toHaveProperty("reasoning_effort");
  });

  test("classifies length and retries empty Chat Completions content", async () => {
    const lengthResponse = vi.fn(async () => new Response(JSON.stringify({
      choices: [{ finish_reason: "length", message: { content: "" } }],
      usage: { completion_tokens: 4096 },
    }), { status: 200 }));
    const lengthError = await generateOpenAiChatExpansion({
      apiKey: "key",
      model: "deepseek-v4-flash",
      query: "query",
      fetchImpl: lengthResponse,
    }).catch(value => value as OpenAiGenerationError);
    expect(lengthError.code).toBe("incomplete_max_tokens");

    const emptyThenValid = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        choices: [{ finish_reason: "stop", message: { content: "" } }],
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        choices: [{
          finish_reason: "stop",
          message: { content: JSON.stringify({ expansions: [] }) },
        }],
      }), { status: 200 }));
    await generateOpenAiChatExpansion({
      apiKey: "key",
      model: "deepseek-v4-flash",
      query: "query",
      maxAttempts: 2,
      retryDelayMs: 0,
      fetchImpl: emptyThenValid,
    });
    expect(emptyThenValid).toHaveBeenCalledTimes(2);
  });
});
