import { createHash } from "node:crypto";
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
  if (parsed.diagnostics.length > 0) {
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

/** Refuse to resume a partial produced by a different generator or query set. */
function assertResumableIdentity(
  partialPath: string,
  metaPath: string,
  identity: Record<string, string>,
): void {
  if (readFileSync(partialPath, "utf8").trim().length === 0) {
    writeFileSync(metaPath, `${JSON.stringify({ identity }, null, 2)}\n`, "utf8");
    return;
  }
  if (!existsSync(metaPath)) {
    throw new Error(`Partial expansion artifact has no identity record; rerun with --force: ${partialPath}`);
  }
  let recorded: Record<string, string>;
  try {
    recorded = (JSON.parse(readFileSync(metaPath, "utf8")) as { identity: Record<string, string> }).identity;
    if (typeof recorded !== "object" || recorded === null) throw new Error("missing identity");
  } catch {
    throw new Error(`Unreadable partial identity record; rerun with --force: ${metaPath}`);
  }
  const keys = [...new Set([...Object.keys(recorded), ...Object.keys(identity)])].sort();
  const changed = keys.filter(key => recorded[key] !== identity[key]);
  if (changed.length > 0) {
    throw new Error(
      `Partial expansion artifact was produced with a different ${changed.join(", ")}; `
      + `rerun with --force: ${partialPath}`,
    );
  }
}

export async function generateExpansionArtifact(options: {
  benchmarkDir: string;
  runName: string;
  generateRaw: (query: string) => Promise<string>;
  /** Everything that determines generated output (model, prompt, ...); a partial is resumed only on an exact match. */
  identity: Record<string, string>;
  outputPath?: string;
  force?: boolean;
  onProgress?: (progress: ExpansionGenerationProgress) => void;
}): Promise<ExpansionGenerationSummary> {
  const root = resolve(options.benchmarkDir);
  const runName = validateBenchmarkRunName(options.runName);
  const queriesBytes = readFileSync(join(root, "queries.jsonl"));
  const queries = parseQueriesJsonl(queriesBytes.toString("utf8"));
  const identity = {
    ...options.identity,
    queries_sha256: createHash("sha256").update(queriesBytes).digest("hex"),
  };
  const outputPath = resolve(
    options.outputPath ?? join(root, "expansions", `${runName}.jsonl`),
  );
  const partialPath = `${outputPath}.partial`;
  const metaPath = `${partialPath}.meta.json`;
  mkdirSync(dirname(outputPath), { recursive: true });

  if (existsSync(outputPath)) {
    if (!options.force) {
      throw new Error(`Expansion artifact already exists: ${outputPath}`);
    }
    rmSync(outputPath);
  }
  if (options.force) {
    rmSync(partialPath, { force: true });
    rmSync(metaPath, { force: true });
  }
  if (!existsSync(partialPath)) writeFileSync(partialPath, "", "utf8");
  assertResumableIdentity(partialPath, metaPath, identity);

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
  rmSync(metaPath, { force: true });

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
