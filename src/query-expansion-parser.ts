/** Pure parsing and acceptance helpers for generated query expansions. */

export type GeneratedExpansionType = "lex" | "vec" | "hyde";
export type GeneratedExpansion = [GeneratedExpansionType, string];

export type GeneratedExpansionDiagnostic = {
  code:
    | "RUNTIME_EMPTY_OUTPUT"
    | "RUNTIME_INVALID_LINE"
    | "RUNTIME_EMPTY_TEXT";
  line: number;
  text: string;
};

export type ParsedGeneratedExpansionLines = {
  output: GeneratedExpansion[];
  diagnostics: GeneratedExpansionDiagnostic[];
};

const CANONICAL_LINE = /^(lex|vec|hyde): ([^\r\n]+)$/;

/**
 * Parse every line independently.  Production may keep valid lines, while a
 * stricter consumers can reject the response when diagnostics are non-empty.
 */
export function parseGeneratedExpansionLines(
  rawOutput: string,
): ParsedGeneratedExpansionLines {
  const trimmed = rawOutput.trim();
  if (trimmed.length === 0) {
    return {
      output: [],
      diagnostics: [{ code: "RUNTIME_EMPTY_OUTPUT", line: 0, text: "" }],
    };
  }

  const output: GeneratedExpansion[] = [];
  const diagnostics: GeneratedExpansionDiagnostic[] = [];
  const lines = trimmed.split("\n");
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!.replace(/\r$/, "");
    const canonicalMatch = CANONICAL_LINE.exec(line);

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
      continue;
    }
    if (!canonicalMatch) {
      diagnostics.push({
        code: "RUNTIME_INVALID_LINE",
        line: index + 1,
        text: line,
      });
    }
    output.push([type, text]);
  }

  return { output, diagnostics };
}

export function defaultExpansionFallback(query: string): GeneratedExpansion[] {
  return [
    ["hyde", `Information about ${query}`],
    ["lex", query],
    ["vec", query],
  ];
}
