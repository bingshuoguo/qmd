import { openDatabase } from "../db.js";
import type { Database } from "../db.js";
import { spawn as nodeSpawn } from "child_process";
import { fileURLToPath } from "url";
import { basename, dirname, join as pathJoin, relative as relativePath, resolve as pathResolve } from "path";
import { parseArgs } from "util";
import { readFileSync, readdirSync, realpathSync, statSync, existsSync, unlinkSync, writeFileSync, openSync, closeSync, mkdirSync } from "fs";
import {
  getPwd,
  getRealPath,
  homedir,
  resolve,
  enableProductionMode,
  listCollections,
  removeCollection,
  renameCollection,
  findSimilarFiles,
  clearAllEmbeddings,
  insertEmbedding,
  getStatus,
  extractTitle,
  getCacheKey,
  getCachedResult,
  setCachedResult,
  parseVirtualPath,
  resolveVirtualPath,
  toVirtualPath,
  findActiveDocument,
  deleteLLMCache,
  deleteInactiveDocuments,
  cleanupOrphanedVectors,
  vacuumDatabase,
  getCollectionsWithoutContext,
  getTopLevelPathsWithoutContext,
  handelize,
  DEFAULT_GLOB,
  DEFAULT_MULTI_GET_MAX_BYTES,
  createStore,
  type ReindexResult,
  type ChunkStrategy,
} from "../store.js";
import { disposeDefaultLlamaCpp, pullModels, DEFAULT_MODEL_CACHE_DIR, resolveEmbedModel, resolveGenerateModel, resolveRerankModel } from "../llm.js";
import {
  formatSearchResults,
  formatDocuments,
  type OutputFormat,
} from "./formatter.js";
import {
  getCollection as getCollectionFromYaml,
  listCollections as yamlListCollections,
  removeCollection as yamlRemoveCollectionFn,
  renameCollection as yamlRenameCollectionFn,
  listAllContexts,
  setConfigIndexName,
  setConfigSource,
  findLocalConfigPath,
  getLocalDbPath,
  getConfigPath,
  configExists,
} from "../collections.js";
import {
  getDb,
  closeDb,
  getDbPath,
  setStoreDbPathOverride,
  setIndexName,
  resolveEmbedModelForCli,
  resolveModelsForCli,
} from "./context.js";
import {
  c,
  cursor,
  formatBytes,
} from "./term.js";
import {
  type OutputOptions,
} from "./output.js";
import {
  showHelp,
  showSkillsHelp,
  showSkillHelp,
  showCollectionHelp,
  printContextUsage,
  printDoctorHint,
  showVersion,
} from "./help.js";
import { runSkillsCommand, showSkill, installSkill, outputSkillsJson } from "./commands/skills.js";
import { showStatus, showDoctor } from "./commands/doctor.js";
import { getDocument, multiGet, listFiles } from "./commands/docs.js";
import { search, vectorSearch, querySearch, resolveCollectionFilter } from "./commands/search.js";
import {
  updateCollections,
  vectorIndex,
  parseEmbedBatchOption,
  parseChunkStrategy,
  parseEmbedTimeoutOption,
} from "./commands/indexing.js";
import {
  initLocalIndex,
  contextAdd,
  contextList,
  contextRemove,
  collectionList,
  collectionAdd,
  collectionRemove,
  collectionRename,
} from "./commands/collections.js";

// I4: these remain importable from src/cli/qmd.ts for existing consumers
// (test/cli.test.ts, cli-exit-lifecycle.test.ts) after moving to context.ts.
export {
  resolveEmbedModelForCli,
  resolveGenerateModelForCli,
  resolveRerankModelForCli,
} from "./context.js";
export { buildEditorUri, termLink } from "./output.js";

// NOTE: enableProductionMode() is intentionally NOT called at module scope here.
// Importing this module for its exports (e.g. buildEditorUri, termLink from
// test/cli.test.ts) must not flip the global production flag, as that leaks
// into unrelated tests that rely on the default (development) database path
// resolution. The flag is flipped inside the CLI's main-module guard below so
// it only fires when qmd is actually invoked as a script.

type CliLifecycleWritable = {
  write(chunk: string | Uint8Array, callback?: (error?: Error | null) => void): boolean;
};

type FinishSuccessfulCliCommandOptions = {
  command: string;
  format?: OutputFormat;
  cleanup?: () => Promise<void>;
  exit?: (code: number) => void;
  stdout?: CliLifecycleWritable;
  stderr?: CliLifecycleWritable;
};

async function flushWritable(stream: CliLifecycleWritable): Promise<void> {
  await new Promise<void>((resolve) => {
    stream.write("", () => resolve());
  });
}

/**
 * Finish a successful CLI command after output has been flushed.
 *
 * We deliberately do NOT call `process.exit(0)`. `process.exit()` skips
 * Node's `beforeExit` event, and node-llama-cpp registers a `beforeExit` hook
 * that auto-disposes its native handles. On darwin, without that hook firing,
 * libggml-metal's static `ggml_metal_device` destructor asserts on a
 * non-empty residency-set collection during `__cxa_finalize_ranges` and
 * dumps a multi-kB backtrace (upstream ggml-org/llama.cpp#22593, fix open as
 * PR #22595). Empirically, even with explicit `disposeDefaultLlamaCpp()` the
 * direct `process.exit(0)` path still trips the assertion — letting the
 * event loop drain naturally is what actually clears the rsets.
 *
 * So: set `process.exitCode = 0` and return. The main module finishes, the
 * event loop drains, `beforeExit` fires, native resources tear down in
 * order, and the process exits cleanly. The `GGML_METAL_NO_RESIDENCY=1` env
 * var that `bin/qmd` exports is a defense-in-depth safety net for paths
 * that still call `process.exit()` after loading the native binding
 * (signal handlers, error paths, `bun test`).
 *
 * If the caller passes an explicit `exit` for testability, we honor it —
 * the lifecycle tests verify the legacy flush → cleanup → exit ordering.
 * Production callers must not pass `exit`.
 */
export async function finishSuccessfulCliCommand(options: FinishSuccessfulCliCommandOptions): Promise<void> {
  const stderr = options.stderr ?? process.stderr;

  await flushWritable(options.stdout ?? process.stdout);

  try {
    await (options.cleanup ?? disposeDefaultLlamaCpp)();
  } catch (error) {
    stderr.write(
      `QMD Warning: cleanup after successful output failed (${error instanceof Error ? error.message : String(error)}); exiting 0 because command output completed.\n`
    );
  }
  await flushWritable(stderr);

  if (options.exit) {
    options.exit(0);
    return;
  }

  process.exitCode = 0;
}


// Parse CLI arguments using util.parseArgs
function parseCLI() {
  const { values, positionals } = parseArgs({
    args: process.argv.slice(2), // Skip node and script path
    options: {
      // Global options
      index: {
        type: "string",
      },
      context: {
        type: "string",
      },
      help: { type: "boolean", short: "h" },
      version: { type: "boolean", short: "v" },
      skill: { type: "boolean" },
      global: { type: "boolean" },
      yes: { type: "boolean" },
      // Search options
      n: { type: "string" },
      "min-score": { type: "string" },
      all: { type: "boolean" },
      full: { type: "boolean" },
      format: { type: "string" },          // preferred: --format cli|json|csv|md|xml|files
      // Legacy boolean format aliases. Kept working for back-compat but
      // omitted from the documented help; prefer `--format <kind>`.
      csv: { type: "boolean" },
      md: { type: "boolean" },
      xml: { type: "boolean" },
      files: { type: "boolean" },
      json: { type: "boolean" },
      explain: { type: "boolean" },
      collection: { type: "string", short: "c", multiple: true },  // Filter by collection(s)
      // Collection options
      name: { type: "string" },  // collection name
      mask: { type: "string" },  // glob pattern
      // Embed options
      force: { type: "boolean", short: "f" },
      "max-docs-per-batch": { type: "string" },
      "max-batch-mb": { type: "string" },
      timeout: { type: "string" },  // embed session cap in minutes (0 = no limit; default 30)
      // Update options
      pull: { type: "boolean" },  // git pull before update
      refresh: { type: "boolean" },
      // Get options
      l: { type: "string" },  // max lines
      from: { type: "string" },  // start line
      "max-bytes": { type: "string" },  // max bytes for multi-get
      "line-numbers": { type: "boolean" },  // add line numbers to output (search; default on for get/multi-get)
      "no-line-numbers": { type: "boolean" },  // disable line numbers for get/multi-get
      "full-path": { type: "boolean" },  // show on-disk paths instead of qmd:// (get/multi-get/search/query)
      // Query options
      "candidate-limit": { type: "string", short: "C" },
      "no-rerank": { type: "boolean", default: false },
      "no-gpu": { type: "boolean", default: false },
      intent: { type: "string" },
      // Benchmark v2 options
      run: { type: "string" },
      model: { type: "string" },
      only: { type: "string" },
      "check-index": { type: "boolean" },
      // Chunking options
      "chunk-strategy": { type: "string" },  // "regex" (default) or "auto" (AST for code files)
      // MCP HTTP transport options
      http: { type: "boolean" },
      daemon: { type: "boolean" },
      port: { type: "string" },
      host: { type: "string" },
    },
    allowPositionals: true,
    strict: false, // Allow unknown options to pass through
  });

  if (values["no-gpu"]) {
    process.env.QMD_FORCE_CPU = "1";
  }

  // Select index name (default: "index"). If no explicit --index is supplied,
  // a project-local .qmd/index.yaml overrides the global config/cache paths.
  const indexName = values.index as string | undefined;
  if (indexName) {
    setIndexName(indexName);
    setConfigIndexName(indexName);
    setConfigSource();
  } else {
    const localConfigPath = findLocalConfigPath();
    if (localConfigPath) {
      setConfigSource({ configPath: localConfigPath });
      setStoreDbPathOverride(getLocalDbPath(localConfigPath));
      closeDb();
    } else {
      setConfigSource();
    }
  }

  // Determine output format. Prefer --format <kind>; fall back to the
  // legacy boolean aliases (--csv/--md/--xml/--files/--json) which remain
  // wired up for back-compat but are no longer documented.
  let format: OutputFormat = "cli";
  const rawFormat = typeof values.format === "string" ? values.format.toLowerCase().trim() : "";
  const VALID_FORMATS: ReadonlyArray<OutputFormat> = ["cli", "json", "csv", "md", "xml", "files"];
  if (rawFormat) {
    if ((VALID_FORMATS as ReadonlyArray<string>).includes(rawFormat)) {
      format = rawFormat as OutputFormat;
    } else {
      console.error(`Unknown --format value: ${values.format}`);
      console.error(`Valid: ${VALID_FORMATS.join(", ")}`);
      process.exit(1);
    }
  } else if (values.csv) format = "csv";
  else if (values.md) format = "md";
  else if (values.xml) format = "xml";
  else if (values.files) format = "files";
  else if (values.json) format = "json";

  // Default limit: 20 for --files/--json, 5 otherwise
  // --all means return all results (use very large limit)
  const defaultLimit = (format === "files" || format === "json") ? 20 : 5;
  const isAll = !!values.all;

  const opts: OutputOptions = {
    format,
    full: !!values.full,
    limit: isAll ? 100000 : (values.n ? parseInt(String(values.n), 10) || defaultLimit : defaultLimit),
    minScore: values["min-score"] ? parseFloat(String(values["min-score"])) || 0 : 0,
    all: isAll,
    collection: values.collection as string[] | undefined,
    lineNumbers: !!values["line-numbers"],
    candidateLimit: values["candidate-limit"] ? parseInt(String(values["candidate-limit"]), 10) : undefined,
    skipRerank: !!values["no-rerank"],
    explain: !!values.explain,
    intent: values.intent as string | undefined,
    chunkStrategy: parseChunkStrategy(values["chunk-strategy"]),
    fullPath: !!values["full-path"],
  };

  return {
    command: positionals[0] || "",
    args: positionals.slice(1),
    query: positionals.slice(1).join(" "),
    opts,
    values,
  };
}

function exitWithError(error: unknown, code = 1): never {
  console.error(error instanceof Error ? error.message : String(error));
  printDoctorHint();
  process.exit(code);
}

// Main CLI - only run if this is the main module
const __filename = fileURLToPath(import.meta.url);
const argv1 = process.argv[1];
const isMain = argv1 === __filename
  || argv1?.endsWith("/qmd.ts")
  || argv1?.endsWith("/qmd.js")
  || (argv1 != null && realpathSync(argv1) === __filename);
if (isMain) {
  // Flip to production mode only when this module is executed as the CLI
  // entrypoint, not when imported for its exports. Tests must set INDEX_PATH
  // or use createStore() with an explicit path.
  enableProductionMode();

  const cli = parseCLI();

  if (cli.values.version) {
    await showVersion();
    process.exit(0);
  }

  if (cli.values.skill) {
    showSkill();
    process.exit(0);
  }

  if (cli.values.help && cli.command === "skill") {
    showSkillHelp();
    process.exit(0);
  }

  if (!cli.command || cli.values.help) {
    showHelp();
    process.exit(cli.values.help ? 0 : 1);
  }

  switch (cli.command) {
    case "context": {
      const subcommand = cli.args[0];
      if (!subcommand) {
        printContextUsage();
        process.exit(1);
      }

      switch (subcommand) {
        case "add": {
          if (cli.args.length < 2) {
            console.error("Usage: qmd context add [path] \"text\"");
            console.error("");
            console.error("Examples:");
            console.error("  qmd context add \"Context for current directory\"");
            console.error("  qmd context add . \"Context for current directory\"");
            console.error("  qmd context add /subfolder \"Context for subfolder\"");
            console.error("  qmd context add / \"Global context for all collections\"");
            console.error("");
            console.error("  Using virtual paths:");
            console.error("  qmd context add qmd://journals/ \"Context for entire journals collection\"");
            console.error("  qmd context add qmd://journals/2024 \"Context for 2024 journals\"");
            process.exit(1);
          }

          let pathArg: string | undefined;
          let contextText: string;

          // Check if first arg looks like a path or if it's the context text
          const firstArg = cli.args[1] || '';
          const secondArg = cli.args[2];

          if (secondArg) {
            // Two args: path + context
            pathArg = firstArg;
            contextText = cli.args.slice(2).join(" ");
          } else {
            // One arg: context only (use current directory)
            pathArg = undefined;
            contextText = firstArg;
          }

          await contextAdd(pathArg, contextText);
          break;
        }

        case "list": {
          contextList();
          break;
        }

        case "rm":
        case "remove": {
          if (cli.args.length < 2 || !cli.args[1]) {
            console.error("Usage: qmd context rm <path>");
            console.error("Examples:");
            console.error("  qmd context rm /");
            console.error("  qmd context rm qmd://journals/2024");
            process.exit(1);
          }
          contextRemove(cli.args[1]);
          break;
        }

        default:
          console.error(`Unknown subcommand: ${subcommand}`);
          console.error("Available: add, list, rm");
          process.exit(1);
      }
      break;
    }

    case "get": {
      if (!cli.args[0]) {
        console.error("Usage: qmd get <filepath>[:from[:count]] [--from <line>] [-l <lines>] [--no-line-numbers] [--full-path]");
        process.exit(1);
      }
      const fromLine = cli.values.from ? parseInt(cli.values.from as string, 10) : undefined;
      const maxLines = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
      // Line numbers default ON for get; opt out with --no-line-numbers.
      const getLineNumbers = !cli.values["no-line-numbers"];
      getDocument(cli.args[0], fromLine, maxLines, getLineNumbers, !!cli.values["full-path"]);
      break;
    }

    case "multi-get": {
      if (!cli.args[0]) {
        console.error("Usage: qmd multi-get <pattern> [-l <lines>] [--max-bytes <bytes>] [--no-line-numbers] [--full-path] [--format json|csv|md|xml|files]");
        console.error("  pattern: glob (e.g., 'journals/2025-05*.md') or comma-separated list");
        process.exit(1);
      }
      const maxLinesMulti = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
      const maxBytes = cli.values["max-bytes"] ? parseInt(cli.values["max-bytes"] as string, 10) : DEFAULT_MULTI_GET_MAX_BYTES;
      // Line numbers default ON for multi-get; opt out with --no-line-numbers.
      const mgLineNumbers = !cli.values["no-line-numbers"];
      multiGet(cli.args[0], maxLinesMulti, maxBytes, cli.opts.format, mgLineNumbers, !!cli.values["full-path"]);
      break;
    }

    case "ls": {
      listFiles(cli.args[0]);
      break;
    }

    case "collection": {
      const subcommand = cli.args[0];
      switch (subcommand) {
        case "list": {
          collectionList();
          break;
        }

        case "add": {
          const pwd = cli.args[1] || getPwd();
          const resolvedPwd = pwd === '.' ? getPwd() : getRealPath(resolve(pwd));
          const globPattern = cli.values.mask as string || DEFAULT_GLOB;
          const name = cli.values.name as string | undefined;

          await collectionAdd(resolvedPwd, globPattern, name);
          break;
        }

        case "remove":
        case "rm": {
          if (!cli.args[1]) {
            console.error("Usage: qmd collection remove <name>");
            console.error("  Use 'qmd collection list' to see available collections");
            process.exit(1);
          }
          collectionRemove(cli.args[1]);
          break;
        }

        case "rename":
        case "mv": {
          if (!cli.args[1] || !cli.args[2]) {
            console.error("Usage: qmd collection rename <old-name> <new-name>");
            console.error("  Use 'qmd collection list' to see available collections");
            process.exit(1);
          }
          collectionRename(cli.args[1], cli.args[2]);
          break;
        }

        case "set-update":
        case "update-cmd": {
          const name = cli.args[1];
          const cmd = cli.args.slice(2).join(' ') || null;
          if (!name) {
            console.error("Usage: qmd collection update-cmd <name> [command]");
            console.error("  Set the command to run before indexing (e.g., 'git pull')");
            console.error("  Omit command to clear it");
            process.exit(1);
          }
          const { updateCollectionSettings, getCollection } = await import("../collections.js");
          const col = getCollection(name);
          if (!col) {
            console.error(`Collection not found: ${name}`);
            process.exit(1);
          }
          updateCollectionSettings(name, { update: cmd });
          if (cmd) {
            console.log(`✓ Set update command for '${name}': ${cmd}`);
          } else {
            console.log(`✓ Cleared update command for '${name}'`);
          }
          break;
        }

        case "include":
        case "exclude": {
          const name = cli.args[1];
          if (!name) {
            console.error(`Usage: qmd collection ${subcommand} <name>`);
            console.error(`  ${subcommand === 'include' ? 'Include' : 'Exclude'} collection in default queries`);
            process.exit(1);
          }
          const { updateCollectionSettings, getCollection } = await import("../collections.js");
          const col = getCollection(name);
          if (!col) {
            console.error(`Collection not found: ${name}`);
            process.exit(1);
          }
          const include = subcommand === 'include';
          updateCollectionSettings(name, { includeByDefault: include });
          console.log(`✓ Collection '${name}' ${include ? 'included in' : 'excluded from'} default queries`);
          break;
        }

        case "show":
        case "info": {
          const name = cli.args[1];
          if (!name) {
            console.error("Usage: qmd collection show <name>");
            process.exit(1);
          }
          const { getCollection } = await import("../collections.js");
          const col = getCollection(name);
          if (!col) {
            console.error(`Collection not found: ${name}`);
            process.exit(1);
          }
          console.log(`Collection: ${name}`);
          console.log(`  Path:     ${col.path}`);
          console.log(`  Pattern:  ${col.pattern}`);
          console.log(`  Include:  ${col.includeByDefault !== false ? 'yes (default)' : 'no'}`);
          if (col.update) {
            console.log(`  Update:   ${col.update}`);
          }
          if (col.context) {
            const ctxCount = Object.keys(col.context).length;
            console.log(`  Contexts: ${ctxCount}`);
          }
          break;
        }

        case "help":
        case undefined: {
          showCollectionHelp();
          process.exit(0);
        }

        default:
          console.error(`Unknown subcommand: ${subcommand}`);
          console.error("Run 'qmd collection help' for usage");
          printDoctorHint();
          process.exit(1);
      }
      break;
    }

    case "init":
      try {
        initLocalIndex();
      } catch (error) {
        exitWithError(error);
      }
      break;

    case "status":
      await showStatus();
      break;

    case "doctor":
      await showDoctor();
      break;

    case "update":
      await updateCollections();
      break;

    case "embed":
      try {
        const maxDocsPerBatch = parseEmbedBatchOption("maxDocsPerBatch", cli.values["max-docs-per-batch"]);
        const maxBatchMb = parseEmbedBatchOption("maxBatchBytes", cli.values["max-batch-mb"]);
        const embedChunkStrategy = parseChunkStrategy(cli.values["chunk-strategy"]);
        const embedMaxDurationMs = parseEmbedTimeoutOption(cli.values["timeout"]);
        // Validate -c against configured collections before dispatching, so a
        // typo errors with "Collection not found: X" instead of silently
        // reporting success because no pending docs match a nonexistent name.
        // embed operates on a single collection; only the first value is used.
        const embedValidatedCollections = resolveCollectionFilter(cli.opts.collection, false);
        const embedCollection = embedValidatedCollections[0];
        await vectorIndex(resolveEmbedModelForCli(), !!cli.values.force, {
          maxDocsPerBatch,
          maxBatchBytes: maxBatchMb === undefined ? undefined : maxBatchMb * 1024 * 1024,
          chunkStrategy: embedChunkStrategy,
          collection: embedCollection,
          maxDurationMs: embedMaxDurationMs,
        });
      } catch (error) {
        exitWithError(error);
      }
      break;

    case "pull": {
      const refresh = cli.values.refresh === undefined ? false : Boolean(cli.values.refresh);
      const activeModels = resolveModelsForCli();
      const models = [
        activeModels.embed,
        activeModels.generate,
        activeModels.rerank,
      ];
      console.log(`${c.bold}Pulling models${c.reset}`);
      const results = await pullModels(models, {
        refresh,
        cacheDir: DEFAULT_MODEL_CACHE_DIR,
      });
      for (const result of results) {
        const size = formatBytes(result.sizeBytes);
        const note = result.refreshed ? "refreshed" : "cached/checked";
        console.log(`- ${result.model} -> ${result.path} (${size}, ${note})`);
      }
      break;
    }

    case "search":
      if (!cli.query) {
        console.error("Usage: qmd search [options] <query>");
        process.exit(1);
      }
      search(cli.query, cli.opts);
      break;

    case "vsearch":
    case "vector-search": // undocumented alias
      if (!cli.query) {
        console.error("Usage: qmd vsearch [options] <query>");
        process.exit(1);
      }
      // Default min-score for vector search is 0.3
      if (!cli.values["min-score"]) {
        cli.opts.minScore = 0.3;
      }
      await vectorSearch(cli.query, cli.opts);
      break;

    case "query":
    case "deep-search": // undocumented alias
      if (!cli.query) {
        console.error("Usage: qmd query [options] <query>");
        process.exit(1);
      }
      await querySearch(cli.query, cli.opts);
      break;

    case "bench": {
      const fixturePath = cli.args[0];
      if (!fixturePath) {
        console.error("Usage: qmd bench <fixture.json> [--json] [-c collection]");
        console.error("       qmd bench <benchmark-dir> --run <name> [--model ID] [--only lex|vec|hyde]");
        console.error("       qmd bench <benchmark-dir> --check-index");
        console.error("");
        console.error("Run legacy fixture or canonical qrels benchmarks.");
        console.error("See src/bench/fixtures/example.json for the fixture format.");
        process.exit(1);
      }
      const runValue = typeof cli.values.run === "string"
        ? cli.values.run
        : undefined;
      if (runValue !== undefined) {
        const { validateBenchmarkRunName } = await import("../bench/run-name.js");
        try {
          validateBenchmarkRunName(runValue);
        } catch (error) {
          console.error(error instanceof Error ? error.message : String(error));
          process.exit(1);
        }
      }
      const onlyValue = cli.values.only;
      if (
        onlyValue !== undefined
        && onlyValue !== "lex"
        && onlyValue !== "vec"
        && onlyValue !== "hyde"
      ) {
        console.error("--only must be lex, vec, or hyde");
        process.exit(1);
      }
      const modelValue = typeof cli.values.model === "string"
        ? cli.values.model
        : undefined;
      const { runBenchmarkCommand, writeBenchmarkIndexManifest } = await import("../bench/bench.js");
      const benchCollection = cli.opts.collection;
      const common = {
        dbPath: getDbPath(),
        configPath: configExists() ? getConfigPath() : undefined,
      };
      if (cli.values["check-index"]) {
        if (runValue || modelValue || onlyValue) {
          console.error("--check-index cannot be combined with --run, --model, or --only");
          process.exit(1);
        }
        const manifest = await writeBenchmarkIndexManifest(fixturePath, common);
        console.log(JSON.stringify(manifest, null, 2));
      } else {
        await runBenchmarkCommand(fixturePath, {
          json: !!cli.values.json,
          collection: Array.isArray(benchCollection) ? benchCollection[0] : benchCollection,
          ...common,
          run: runValue,
          model: modelValue,
          only: onlyValue,
        });
      }
      break;
    }

    case "mcp": {
      const sub = cli.args[0]; // stop | status | undefined

      // Cache dir for PID/log files — same dir as the index
      const cacheDir = process.env.XDG_CACHE_HOME
        ? resolve(process.env.XDG_CACHE_HOME, "qmd")
        : resolve(homedir(), ".cache", "qmd");
      const pidPath = resolve(cacheDir, "mcp.pid");

      // Subcommands take priority over flags
      if (sub === "stop") {
        if (!existsSync(pidPath)) {
          console.log("Not running (no PID file).");
          process.exit(0);
        }
        const pid = parseInt(readFileSync(pidPath, "utf-8").trim());
        try {
          process.kill(pid, 0); // alive?
          process.kill(pid, "SIGTERM");
          unlinkSync(pidPath);
          console.log(`Stopped QMD MCP server (PID ${pid}).`);
        } catch {
          unlinkSync(pidPath);
          console.log("Cleaned up stale PID file (server was not running).");
        }
        process.exit(0);
      }

      if (cli.values.http) {
        const port = Number(cli.values.port) || 8181;
        // --host overrides the default localhost bind; QMD_HOST env is the
        // fallback (resolved in startMcpHttpServer). Use "0.0.0.0" to accept
        // off-host connections, e.g. a container liveness probe.
        const host = cli.values.host ? String(cli.values.host) : undefined;

        if (cli.values.daemon) {
          // Guard: check if already running
          if (existsSync(pidPath)) {
            const existingPid = parseInt(readFileSync(pidPath, "utf-8").trim());
            try {
              process.kill(existingPid, 0); // alive?
              console.error(`Already running (PID ${existingPid}). Run 'qmd mcp stop' first.`);
              process.exit(1);
            } catch {
              // Stale PID file — continue
            }
          }

          mkdirSync(cacheDir, { recursive: true });
          const logPath = resolve(cacheDir, "mcp.log");
          const logFd = openSync(logPath, "w"); // truncate — fresh log per daemon run
          const selfPath = fileURLToPath(import.meta.url);
          const indexArgs = cli.values.index ? ["--index", String(cli.values.index)] : [];
          const hostArgs = host ? ["--host", host] : [];
          const spawnArgs = selfPath.endsWith(".ts")
            ? ["--import", pathJoin(dirname(selfPath), "..", "..", "node_modules", "tsx", "dist", "esm", "index.mjs"), selfPath, ...indexArgs, "mcp", "--http", "--port", String(port), ...hostArgs]
            : [selfPath, ...indexArgs, "mcp", "--http", "--port", String(port), ...hostArgs];
          const child = nodeSpawn(process.execPath, spawnArgs, {
            stdio: ["ignore", logFd, logFd],
            detached: true,
          });
          child.unref();
          closeSync(logFd); // parent's copy; child inherited the fd

          writeFileSync(pidPath, String(child.pid));
          console.log(`Started on http://${host ?? "localhost"}:${port}/mcp (PID ${child.pid})`);
          console.log(`Logs: ${logPath}`);
          process.exit(0);
        }

        // Foreground HTTP mode — remove top-level cursor handlers so the
        // async cleanup handlers in startMcpHttpServer actually run.
        process.removeAllListeners("SIGTERM");
        process.removeAllListeners("SIGINT");
        const { startMcpHttpServer } = await import("../mcp/server.js");
        try {
          await startMcpHttpServer(port, { dbPath: getDbPath(), host });
        } catch (e: unknown) {
          if (typeof e === "object" && e !== null && "code" in e && e.code === "EADDRINUSE") {
            console.error(`Port ${port} already in use. Try a different port with --port.`);
            process.exit(1);
          }
          throw e;
        }
      } else {
        // Default: stdio transport
        const { startMcpServer } = await import("../mcp/server.js");
        await startMcpServer({ dbPath: getDbPath() });
      }
      break;
    }

    case "skills": {
      try {
        if (cli.values.help || cli.args[0] === "help") {
          showSkillsHelp();
        } else {
          runSkillsCommand(cli.args, Boolean(cli.values.json), Boolean(cli.values.full), Boolean(cli.values.all));
        }
      } catch (error) {
        if (cli.values.json) {
          outputSkillsJson({ success: false, error: error instanceof Error ? error.message : String(error) });
        } else {
          console.error(error instanceof Error ? error.message : String(error));
        }
        process.exit(1);
      }
      break;
    }

    case "skill": {
      const subcommand = cli.args[0];
      switch (subcommand) {
        case "show": {
          showSkill();
          break;
        }

        case "install": {
          try {
            await installSkill(Boolean(cli.values.global), Boolean(cli.values.force), Boolean(cli.values.yes));
          } catch (error) {
            exitWithError(error);
          }
          break;
        }

        case "help":
        case undefined: {
          showSkillHelp();
          process.exit(0);
        }

        default:
          console.error(`Unknown subcommand: ${subcommand}`);
          console.error("Run 'qmd skill help' for usage");
          printDoctorHint();
          process.exit(1);
      }
      break;
    }

    case "cleanup": {
      const db = getDb();

      // 1. Clear llm_cache
      const cacheCount = deleteLLMCache(db);
      console.log(`${c.green}✓${c.reset} Cleared ${cacheCount} cached API responses`);

      // 2. Remove orphaned vectors
      const orphanedVecs = cleanupOrphanedVectors(db);
      if (orphanedVecs > 0) {
        console.log(`${c.green}✓${c.reset} Removed ${orphanedVecs} orphaned embedding chunks`);
      } else {
        console.log(`${c.dim}No orphaned embeddings to remove${c.reset}`);
      }

      // 3. Remove inactive documents
      const inactiveDocs = deleteInactiveDocuments(db);
      if (inactiveDocs > 0) {
        console.log(`${c.green}✓${c.reset} Removed ${inactiveDocs} inactive document records`);
      }

      // 4. Vacuum to reclaim space
      vacuumDatabase(db);
      console.log(`${c.green}✓${c.reset} Database vacuumed`);

      closeDb();
      break;
    }

    default:
      console.error(`Unknown command: ${cli.command}`);
      console.error("Run 'qmd --help' for usage.");
      printDoctorHint();
      process.exit(1);
  }

  if (cli.command !== "mcp") {
    await finishSuccessfulCliCommand({
      command: cli.command,
      format: cli.opts.format,
    });
  }

} // end if (main module)
