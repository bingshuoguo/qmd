import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseGeneratedExpansionLines } from "../src/query-expansion-parser.js";
import { parseProductionExpansion } from "../src/llm.js";

type Fixture = {
  cases: Array<{
    name: string;
    query: string;
    raw_model_output?: string;
    runtime?: {
      output: Array<["lex" | "vec" | "hyde", string]>;
      fallback: boolean;
    };
  }>;
};

const fixturePath = fileURLToPath(new URL(
  "../finetune/fixtures/query-expansion-contract-v1.json",
  import.meta.url,
));
const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as Fixture;

describe("generated expansion line parser", () => {
  test("keeps usable lines and reports bad lines independently", () => {
    expect(parseGeneratedExpansionLines("lex: auth\nnot a typed line"))
      .toEqual({
        output: [["lex", "auth"]],
        diagnostics: [{
          code: "RUNTIME_INVALID_LINE",
          line: 2,
          text: "not a typed line",
        }],
      });
  });

  test("recognizes canonical syntax without imposing it on runtime", () => {
    expect(parseGeneratedExpansionLines("lex:auth")).toEqual({
      output: [["lex", "auth"]],
      diagnostics: [{
        code: "RUNTIME_INVALID_LINE",
        line: 1,
        text: "lex:auth",
      }],
    });
  });

  test("drops empty text and falls back when no usable line remains", () => {
    expect(parseGeneratedExpansionLines("lex:")).toEqual({
      output: [],
      diagnostics: [{ code: "RUNTIME_EMPTY_TEXT", line: 1, text: "lex:" }],
    });
    expect(parseProductionExpansion("内存管理", "lex:")).toEqual({
      output: [
        { type: "hyde", text: "Information about 内存管理" },
        { type: "lex", text: "内存管理" },
        { type: "vec", text: "内存管理" },
      ],
      fallbackUsed: true,
    });
  });

  test("keeps valid synonym expansions and falls back only when no line parses", () => {
    expect(parseProductionExpansion("videography tips", "lex: video production"))
      .toEqual({
        output: [{ type: "lex", text: "video production" }],
        fallbackUsed: false,
      });
    expect(parseProductionExpansion("auth", "not a typed line")).toEqual({
      output: [
        { type: "hyde", text: "Information about auth" },
        { type: "lex", text: "auth" },
        { type: "vec", text: "auth" },
      ],
      fallbackUsed: true,
    });
  });

  test("matches runtime expectations in shared fixtures", () => {
    for (const fixtureCase of fixture.cases.filter(item => item.runtime)) {
      const result = parseProductionExpansion(
        fixtureCase.query,
        fixtureCase.raw_model_output ?? "",
      );

      expect(
        {
          output: result.output.map(item => [item.type, item.text]),
          fallback: result.fallbackUsed,
        },
        fixtureCase.name,
      ).toEqual(fixtureCase.runtime);
    }
  });
});
