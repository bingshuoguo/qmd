/**
 * help.ts - help/usage text, version reporting, and the doctor hint for the
 * qmd CLI. Leaf module: depends only on context.ts (getDbPath) and node
 * builtins.
 */
import { execSync } from "child_process";
import { readFileSync } from "fs";
import { dirname } from "path";
import { fileURLToPath } from "url";
import { resolve } from "../store.js";
import { getDbPath } from "./context.js";

function showHelp(): void {
  console.log("qmd — Quick Markdown Search");
  console.log("");
  console.log("Usage:");
  console.log("  qmd <command> [options]");
  console.log("");
  console.log("Primary commands:");
  console.log("  qmd query <query>             - Hybrid search with auto expansion + reranking (recommended)");
  console.log("  qmd query 'lex:..\\nvec:...'   - Structured query document (you provide lex/vec/hyde lines)");
  console.log("  qmd search <query>            - Full-text BM25 keywords (no LLM)");
  console.log("  qmd vsearch <query>           - Vector similarity only");
  console.log("  qmd get <file>[:from[:count]] - Show a document (line-numbered; #docid in header)");
  console.log("  qmd multi-get <pattern>       - Batch fetch via glob or comma-separated list");
  console.log("  qmd skills list/get/path      - List and retrieve bundled runtime skills");
  console.log("  qmd skill show/install        - Show or install the QMD skill");
  console.log("  qmd mcp                       - Start the MCP server (stdio transport for AI agents)");
  console.log("  qmd bench <fixture-or-dir>    - Run legacy or qrels search quality benchmarks");
  console.log("");
  console.log("Collections & context:");
  console.log("  qmd collection add/list/remove/rename/show   - Manage indexed folders");
  console.log("  qmd context add/list/rm                      - Attach human-written summaries");
  console.log("  qmd ls [collection[/path]]                   - Inspect indexed files");
  console.log("");
  console.log("Maintenance:");
  console.log("  qmd init                      - Create a project-local .qmd index");
  console.log("  qmd status                    - View index + collection health");
  console.log("  qmd update [--pull]           - Re-index collections (optionally git pull first)");
  console.log("  qmd embed [-f] [-c <name>]    - Generate/refresh vector embeddings");
  console.log("    --max-docs-per-batch <n>    - Cap docs loaded into memory per embedding batch");
  console.log("    --max-batch-mb <n>          - Cap UTF-8 MB loaded into memory per embedding batch");
  console.log("    --timeout <minutes>         - Embed session cap in minutes (0 = no limit; default 30)");
  console.log("  qmd cleanup                   - Clear caches, vacuum DB");
  console.log("");
  console.log("Query syntax (qmd query):");
  console.log("  QMD queries are either a single expand query (no prefix) or a multi-line");
  console.log("  document where every line is typed with lex:, vec:, or hyde:. This grammar");
  console.log("  matches the docs in docs/SYNTAX.md and is enforced in the CLI.");
  console.log("");
  const grammar = [
    `query          = expand_query | query_document ;`,
    `expand_query   = text | explicit_expand ;`,
    `explicit_expand= "expand:" text ;`,
    `query_document = [ intent_line ] { typed_line } ;`,
    `intent_line    = "intent:" text newline ;`,
    `typed_line     = type ":" text newline ;`,
    `type           = "lex" | "vec" | "hyde" ;`,
    `text           = quoted_phrase | plain_text ;`,
    `quoted_phrase  = '"' { character } '"' ;`,
    `plain_text     = { character } ;`,
    `newline        = "\\n" ;`,
  ];
  console.log("  Grammar:");
  for (const line of grammar) {
    console.log(`    ${line}`);
  }
  console.log("");
  console.log("  Examples:");
  console.log("    qmd query \"how does auth work\"                # single-line → implicit expand");
  console.log("    qmd query $'lex: CAP theorem\\nvec: consistency'  # typed query document");
  console.log("    qmd query $'lex: \"exact matches\" sports -baseball'  # phrase + negation lex search");
  console.log("    qmd query $'hyde: Hypothetical answer text'       # hyde-only document");
  console.log("");
  console.log("  Constraints:");
  console.log("    - Standalone expand queries cannot mix with typed lines.");
  console.log("    - Query documents allow only lex:, vec:, or hyde: prefixes.");
  console.log("    - Each typed line must be single-line text with balanced quotes.");
  console.log("");
  console.log("AI agents & integrations:");
  console.log("  - Run `qmd mcp` to expose the MCP server (stdio) to agents/IDEs.");
  console.log("  - Run `qmd skills get qmd --full` for version-matched agent instructions.");
  console.log("  - `qmd skill install` installs the QMD skill into ./.agents/skills/qmd.");
  console.log("  - Use `qmd skill install --global` for ~/.agents/skills/qmd.");
  console.log("  - `qmd --skill` is kept as an alias for `qmd skill show`.");
  console.log("  - Advanced: `qmd mcp --http ...` and `qmd mcp --http --daemon` are optional for custom transports.");
  console.log("");
  console.log("Global options:");
  console.log("  --index <name>             - Use a named index (default: index)");
  console.log("  QMD_EDITOR_URI             - Editor link template for clickable TTY search output");
  console.log("");
  console.log("Search options:");
  console.log("  -n <num>                   - Max results (default 5, or 20 for --format files|json)");
  console.log("  --all                      - Return all matches (pair with --min-score)");
  console.log("  --min-score <num>          - Minimum similarity score");
  console.log("  --full                     - Output full document instead of snippet");
  console.log("  -C, --candidate-limit <n>  - Max candidates to rerank (default 40, lower = faster)");
  console.log("  --no-rerank                - Skip LLM reranking (use RRF scores only, much faster on CPU)");
  console.log("  --no-gpu                   - Force CPU mode for llama.cpp operations (same as QMD_FORCE_CPU=1)");
  console.log("  --line-numbers             - Include line numbers (search; get/multi-get are on by default)");
  console.log("  --no-line-numbers          - Disable line numbers for get/multi-get");
  console.log("  --full-path                - Show on-disk paths instead of qmd:// + docid (get/multi-get/search/query)");
  console.log("                                Paths are ./-prefixed when under $PWD, absolute otherwise");
  console.log("  --explain                  - Include retrieval score traces (query, CLI/--format json)");
  console.log("  --format <kind>            - Output format: cli (default) | json | csv | md | xml | files");
  console.log("  -c, --collection <name>    - Filter by one or more collections");
  console.log("");
  console.log("Embed/query options:");
  console.log("  --chunk-strategy <auto|regex> - Chunking mode (default: regex; auto uses AST for code files)");
  console.log("  --timeout <minutes>          - Embed session cap in minutes (0 = no limit; default 30)");
  console.log("");
  console.log("Multi-get options:");
  console.log("  -l <num>                   - Maximum lines per file");
  console.log("  --max-bytes <num>          - Skip files larger than N bytes (default 65536)");
  console.log("  --format <kind>            - Same formats as search");
  console.log("");
  console.log(`Index: ${getDbPath()}`);
}

function showSkillsHelp(): void {
  console.log("Usage: qmd skills <list|get|path> [options]");
  console.log("");
  console.log("Commands:");
  console.log("  list                 List bundled runtime skills");
  console.log("  get <name>           Print a bundled runtime skill");
  console.log("  get <name> --full    Include references/templates/scripts");
  console.log("  get --all            Print all bundled runtime skills");
  console.log("  path [name]          Print runtime skill directory path(s)");
  console.log("");
  console.log("Options:");
  console.log("  --json               Print structured JSON");
}

function printDoctorHint(): void {
  console.error("If qmd still behaves unexpectedly, run 'qmd doctor' for diagnostics.");
}

type PackageJson = {
  version: string;
  dependencies?: Record<string, string>;
  devDependencies?: Record<string, string>;
};

function readPackageJson(): PackageJson {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const pkgPath = resolve(scriptDir, "..", "..", "package.json");
  return JSON.parse(readFileSync(pkgPath, "utf-8"));
}

async function showVersion(): Promise<void> {
  const scriptDir = dirname(fileURLToPath(import.meta.url));
  const pkg = readPackageJson();

  let commit = "";
  try {
    commit = execSync(`git -C ${scriptDir} rev-parse --short HEAD`, { encoding: "utf-8", stdio: ["pipe", "pipe", "pipe"] }).trim();
  } catch {
    // Not a git repo or git not available
  }

  const versionStr = commit ? `${pkg.version} (${commit})` : pkg.version;
  console.log(`qmd ${versionStr}`);
}

export {
  showHelp,
  showSkillsHelp,
  printDoctorHint,
  type PackageJson,
  readPackageJson,
  showVersion,
};
