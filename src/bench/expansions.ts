import {
  appendFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import {
  parseExpansionsJsonl,
  parseQueriesJsonl,
} from "./qrels.js";
import type {
  BenchmarkExpansion,
  BenchmarkExpansionRecord,
  ExpansionStatus,
} from "./types.js";
import { validateBenchmarkRunName } from "./run-name.js";
import {
  parseGeneratedExpansionLines,
} from "../query-expansion-parser.js";

type ParsedGeneratedExpansion = Pick<
  BenchmarkExpansionRecord,
  "status" | "raw_output" | "output" | "fallback_used" | "error"
>;

function formatError(rawOutput: string, error: string): ParsedGeneratedExpansion {
  return {
    status: "format_error",
    raw_output: rawOutput,
    output: [],
    fallback_used: false,
    error,
  };
}

export function parseGeneratedExpansion(
  _query: string,
  rawOutput: string,
): ParsedGeneratedExpansion {
  const parsed = parseGeneratedExpansionLines(rawOutput);
  if (parsed.diagnostics[0]?.code === "RUNTIME_EMPTY_OUTPUT") {
    return formatError(rawOutput, "model returned empty output");
  }
  if (!parsed.canonicalSyntax || parsed.diagnostics.length > 0) {
    const line = parsed.diagnostics[0]?.line ?? 1;
    return formatError(rawOutput, `line ${line} is not a typed expansion`);
  }

  return {
    status: "ok",
    raw_output: rawOutput,
    output: parsed.output as BenchmarkExpansion[],
    fallback_used: false,
    error: null,
  };
}

export type ExpansionGenerationProgress = {
  completed: number;
  total: number;
  qid: string | null;
  elapsed_ms: number;
  last_query_ms: number | null;
  format_errors: number;
  generation_errors: number;
  fallbacks: number;
};

export type ExpansionGenerationSummary = {
  output_path: string;
  total: number;
  resumed: number;
  generated: number;
  expansion_passes: number;
  format_errors: number;
  generation_errors: number;
  fallbacks: number;
};

export async function generateExpansionArtifact(options: {
  benchmarkDir: string;
  runName: string;
  generateRaw: (query: string) => Promise<string>;
  outputPath?: string;
  force?: boolean;
  onProgress?: (progress: ExpansionGenerationProgress) => void;
}): Promise<ExpansionGenerationSummary> {
  const root = resolve(options.benchmarkDir);
  const runName = validateBenchmarkRunName(options.runName);
  const queries = parseQueriesJsonl(
    readFileSync(join(root, "queries.jsonl"), "utf8"),
  );
  const outputPath = resolve(
    options.outputPath ?? join(root, "expansions", `${runName}.jsonl`),
  );
  const partialPath = `${outputPath}.partial`;
  mkdirSync(dirname(outputPath), { recursive: true });

  if (existsSync(outputPath)) {
    if (!options.force) {
      throw new Error(`Expansion artifact already exists: ${outputPath}`);
    }
    rmSync(outputPath);
  }
  if (options.force) rmSync(partialPath, { force: true });
  if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");

  const partialText = readFileSync(partialPath, "utf8");
  const lineCount = partialText.split(/\r?\n/).filter(line => line.length > 0).length;
  if (lineCount > queries.length) {
    throw new Error(`Partial expansion artifact has ${lineCount} rows for ${queries.length} queries`);
  }
  const records = lineCount === 0
    ? []
    : parseExpansionsJsonl(partialText, queries.slice(0, lineCount));
  const resumed = records.length;
  const startedAt = performance.now();

  const report = (qid: string | null, lastQueryMs: number | null): void => {
    if (!options.onProgress) return;
    options.onProgress({
      completed: records.length,
      total: queries.length,
      qid,
      elapsed_ms: performance.now() - startedAt,
      last_query_ms: lastQueryMs,
      format_errors: records.filter(record => record.status === "format_error").length,
      generation_errors: records.filter(record => record.status === "generation_error").length,
      fallbacks: records.filter(record => record.fallback_used).length,
    });
  };
  report(null, null);

  for (const query of queries.slice(resumed)) {
    const queryStartedAt = performance.now();
    let generated: ParsedGeneratedExpansion;
    try {
      generated = parseGeneratedExpansion(query.query, await options.generateRaw(query.query));
    } catch (error) {
      generated = {
        status: "generation_error",
        raw_output: "",
        output: [],
        fallback_used: false,
        error: error instanceof Error ? error.message : String(error),
      };
    }
    const record: BenchmarkExpansionRecord = {
      qid: query.qid,
      query: query.query,
      ...generated,
    };
    records.push(record);
    appendFileSync(partialPath, `${JSON.stringify(record)}\n`, "utf8");
    report(query.qid, performance.now() - queryStartedAt);
  }

  const completedText = readFileSync(partialPath, "utf8");
  parseExpansionsJsonl(completedText, queries);
  renameSync(partialPath, outputPath);

  const count = (status: ExpansionStatus): number =>
    records.filter(record => record.status === status).length;
  return {
    output_path: outputPath,
    total: queries.length,
    resumed,
    generated: queries.length - resumed,
    expansion_passes: count("ok"),
    format_errors: count("format_error"),
    generation_errors: count("generation_error"),
    fallbacks: records.filter(record => record.fallback_used).length,
  };
}
