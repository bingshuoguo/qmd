/** Pure parsing and acceptance helpers for generated query expansions. */

export type GeneratedExpansionType = "lex" | "vec" | "hyde";
export type GeneratedExpansion = [GeneratedExpansionType, string];

/** Bump when production acceptance changes and expansion cache entries must not be reused. */
export const GENERATED_OUTPUT_PARSER_VERSION = "generated-output-parser-v2";

export type GeneratedExpansionDiagnostic = {
  code: "RUNTIME_EMPTY_OUTPUT" | "RUNTIME_INVALID_LINE" | "RUNTIME_EMPTY_TEXT";
  line: number;
  text: string;
};

export type ParsedGeneratedExpansionLines = {
  output: GeneratedExpansion[];
  diagnostics: GeneratedExpansionDiagnostic[];
  canonicalSyntax: boolean;
};

const CANONICAL_LINE = /^(lex|vec|hyde): ([^\r\n]+)$/;

/**
 * Parse every line independently.  Production may keep valid lines, while a
 * stricter consumer can reject the response when diagnostics are non-empty or
 * canonicalSyntax is false.
 */
export function parseGeneratedExpansionLines(
  rawOutput: string,
): ParsedGeneratedExpansionLines {
  const trimmed = rawOutput.trim();
  if (trimmed.length === 0) {
    return {
      output: [],
      diagnostics: [{ code: "RUNTIME_EMPTY_OUTPUT", line: 0, text: "" }],
      canonicalSyntax: false,
    };
  }

  const output: GeneratedExpansion[] = [];
  const diagnostics: GeneratedExpansionDiagnostic[] = [];
  let canonicalSyntax = true;
  const lines = trimmed.split("\n");
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!.replace(/\r$/, "");
    const canonicalMatch = CANONICAL_LINE.exec(line);
    if (!canonicalMatch) canonicalSyntax = false;

    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) {
      diagnostics.push({
        code: "RUNTIME_INVALID_LINE",
        line: index + 1,
        text: line,
      });
      continue;
    }
    const type = line.slice(0, colonIndex).trim();
    if (type !== "lex" && type !== "vec" && type !== "hyde") {
      diagnostics.push({
        code: "RUNTIME_INVALID_LINE",
        line: index + 1,
        text: line,
      });
      continue;
    }
    const text = line.slice(colonIndex + 1).trim();
    if (text.length === 0) {
      diagnostics.push({
        code: "RUNTIME_EMPTY_TEXT",
        line: index + 1,
        text: line,
      });
      // Keep the parsed tuple so the production wrapper preserves its legacy
      // behavior for non-ASCII queries.  Strict consumers reject diagnostics.
      output.push([type, text]);
      continue;
    }
    output.push([type, text]);
  }

  return { output, diagnostics, canonicalSyntax };
}

/**
 * @deprecated Kept as an identity function for source compatibility. Production
 * acceptance no longer uses query-token overlap as a relevance heuristic.
 */
export function filterGeneratedExpansionsForQuery(
  _query: string,
  output: readonly GeneratedExpansion[],
): GeneratedExpansion[] {
  return [...output];
}

export function defaultExpansionFallback(query: string): GeneratedExpansion[] {
  return [
    ["hyde", `Information about ${query}`],
    ["lex", query],
    ["vec", query],
  ];
}
