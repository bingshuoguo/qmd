import { afterEach, describe, expect, test, vi } from "vitest";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  generateExpansionArtifact,
  parseGeneratedExpansion,
} from "../src/bench/expansions.js";
import { validateBenchmarkRunName } from "../src/bench/run-name.js";
import { parseExpansionsJsonl, parseQueriesJsonl } from "../src/bench/qrels.js";

const temporaryDirectories: string[] = [];

afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) {
    rmSync(directory, { recursive: true, force: true });
  }
});

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), "qmd-expansions-"));
  temporaryDirectories.push(root);
  writeFileSync(join(root, "queries.jsonl"), [
    JSON.stringify({ qid: "q1", query: "alpha evidence" }),
    JSON.stringify({ qid: "q2", query: "beta evidence" }),
    "",
  ].join("\n"));
  return root;
}

describe("benchmark model run names", () => {
  test("accepts stable slugs and rejects path-like names", () => {
    expect(validateBenchmarkRunName("upstream-qmd")).toBe("upstream-qmd");
    expect(validateBenchmarkRunName("qwen-1.7b-base")).toBe("qwen-1.7b-base");
    expect(() => validateBenchmarkRunName("../candidate")).toThrow("safe slug");
    expect(() => validateBenchmarkRunName("Candidate Model")).toThrow("safe slug");
  });
});

describe("generated expansion parsing", () => {
  test("preserves valid output order", () => {
    expect(parseGeneratedExpansion(
      "alpha evidence",
      "hyde: alpha hypothetical\nlex: alpha evidence\nvec: evidence about alpha",
    )).toEqual({
      status: "ok",
      raw_output: "hyde: alpha hypothetical\nlex: alpha evidence\nvec: evidence about alpha",
      output: [
        ["hyde", "alpha hypothetical"],
        ["lex", "alpha evidence"],
        ["vec", "evidence about alpha"],
      ],
      fallback_used: false,
      error: null,
    });
  });

  test("records format errors without partial output", () => {
    expect(parseGeneratedExpansion("alpha", "lex: alpha\ninvalid line"))
      .toMatchObject({
        status: "format_error",
        output: [],
        fallback_used: false,
      });
  });

  test("keeps syntactically valid output without literal query overlap", () => {
    expect(parseGeneratedExpansion("alpha", "vec: unrelated"))
      .toMatchObject({
        status: "ok",
        output: [["vec", "unrelated"]],
        fallback_used: false,
      });
  });
});

describe("offline expansion artifact generation", () => {
  test("resumes a validated prefix and atomically finalizes complete output", async () => {
    const root = fixture();
    mkdirSync(join(root, "expansions"));
    const partial = join(root, "expansions", "qwen-base.jsonl.partial");
    writeFileSync(partial, `${JSON.stringify({
      qid: "q1",
      query: "alpha evidence",
      status: "ok",
      raw_output: "lex: alpha evidence",
      output: [["lex", "alpha evidence"]],
      fallback_used: false,
      error: null,
    })}\n`);
    const generateRaw = vi.fn(async (query: string) => `vec: ${query}`);

    const summary = await generateExpansionArtifact({
      benchmarkDir: root,
      runName: "qwen-base",
      generateRaw,
    });

    expect(generateRaw).toHaveBeenCalledOnce();
    expect(generateRaw).toHaveBeenCalledWith("beta evidence");
    expect(summary).toMatchObject({ total: 2, resumed: 1, generated: 1 });
    expect(existsSync(partial)).toBe(false);
    const outputPath = join(root, "expansions", "qwen-base.jsonl");
    const queries = parseQueriesJsonl(readFileSync(join(root, "queries.jsonl"), "utf8"));
    expect(parseExpansionsJsonl(readFileSync(outputPath, "utf8"), queries)).toHaveLength(2);
  });

  test("records generation errors and continues", async () => {
    const root = fixture();
    const summary = await generateExpansionArtifact({
      benchmarkDir: root,
      runName: "candidate-failed",
      generateRaw: async query => {
        if (query.startsWith("alpha")) throw new Error("model unavailable");
        return `lex: ${query}`;
      },
    });
    expect(summary.generation_errors).toBe(1);
    const rows = readFileSync(
      join(root, "expansions", "candidate-failed.jsonl"),
      "utf8",
    ).trim().split("\n").map(line => JSON.parse(line));
    expect(rows[0]).toMatchObject({
      status: "generation_error",
      raw_output: "",
      output: [],
      fallback_used: false,
      error: "model unavailable",
    });
  });
});
