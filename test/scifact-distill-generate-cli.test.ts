import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { afterEach, describe, expect, test } from "vitest";

const execFileAsync = promisify(execFile);
const repositoryRoot = resolve(import.meta.dirname, "..");
const generateScript = join(repositoryRoot, "finetune/benchmarks/distill/generate.ts");
const temporaryRoots: string[] = [];

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

async function fixtureRoot(): Promise<{ root: string; queriesPath: string; splitSha256: string }> {
  const base = await mkdtemp(join(tmpdir(), "qmd-distill-cli-"));
  temporaryRoots.push(base);
  const root = join(base, "distillation");
  await mkdir(root, { recursive: true });
  const queries = `${JSON.stringify({ _id: "q1", text: "Claim alpha is not beta" })}\n`;
  const queriesPath = join(base, "queries.jsonl");
  await writeFile(queriesPath, queries);
  const split = {
    version: "scifact-distill-split-v1",
    dataset: "scifact-v1",
    source: {
      archive_md5: "fixture",
      queries_sha256: sha256(queries),
      train_qrels_sha256: "fixture",
      test_qrels_sha256: "fixture",
    },
    test_benchmark: { benchmark_id: "fixture", benchmark_manifest_sha256: "fixture" },
    algorithm: {
      version: "sha256-qid-v1",
      normalization: "nfkc-lower-whitespace-v1",
      seed: 42,
      validation_count: 161,
    },
    counts: {
      source_train_queries: 1,
      source_test_queries: 0,
      excluded_queries: 0,
      train_queries: 1,
      val_queries: 0,
    },
    exclusions: [],
    train_qids: ["q1"],
    val_qids: [],
  };
  const splitText = `${JSON.stringify(split, null, 2)}\n`;
  await writeFile(join(root, "split.json"), splitText);
  return { root, queriesPath, splitSha256: sha256(splitText) };
}

async function withTeacherServer<T>(run: (endpoint: string, bodies: unknown[]) => Promise<T>): Promise<T> {
  const bodies: unknown[] = [];
  const server = createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", chunk => { body += chunk; });
    request.on("end", () => {
      bodies.push(JSON.parse(body));
      response.writeHead(200, { "Content-Type": "application/json" });
      response.end(JSON.stringify({
        status: "completed",
        output: [{
          type: "message",
          content: [{
            type: "output_text",
            text: JSON.stringify({
              expansions: [
                { type: "lex", query: "alpha beta evidence" },
                { type: "vec", query: "evidence that alpha is not beta" },
              ],
            }),
          }],
        }],
      }));
    });
  });
  await new Promise<void>(resolveListen => server.listen(0, "127.0.0.1", resolveListen));
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("teacher server has no port");
  try {
    return await run(`http://127.0.0.1:${address.port}/v1/responses`, bodies);
  } finally {
    await new Promise<void>((resolveClose, reject) => {
      server.close(error => error ? reject(error) : resolveClose());
    });
  }
}

/**
 * The generator reads a dozen `DISTILL_*` variables and falls back to
 * `OPENAI_API_KEY`. Inheriting the developer's shell let those leak into the
 * child: exporting `DISTILL_SYSTEM_PROMPT_FILE` while a test sets the inline
 * `DISTILL_SYSTEM_PROMPT` made the CLI reject the combination and the test
 * fail for a reason that had nothing to do with the code under test. Strip the
 * whole prefix so each case declares its entire teacher configuration.
 */
function hermeticEnvironment(environment: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  const inherited = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => (
      !key.startsWith("DISTILL_") && key !== "OPENAI_API_KEY"
    )),
  );
  return { ...inherited, ...environment };
}

async function runGenerator(args: string[], environment: NodeJS.ProcessEnv): Promise<void> {
  await execFileAsync(process.execPath, ["--import", "tsx", generateScript, ...args], {
    cwd: repositoryRoot,
    env: hermeticEnvironment(environment),
    timeout: 30_000,
  });
}

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map(path => rm(path, { recursive: true, force: true })));
});

describe("SciFact distill generator CLI", () => {
  test("records resolved prompt and CLI output budget in the manifest", async () => {
    const fixture = await fixtureRoot();
    await withTeacherServer(async (endpoint, bodies) => {
      await runGenerator([
        "--experiment", "prompt-config-test",
        "--model", "gpt-test",
        "--provider", "openai-compatible",
        "--root", fixture.root,
        "--queries", fixture.queriesPath,
        "--max-output-tokens", "2222",
      ], {
        DISTILL_API_KEY: "test-key",
        DISTILL_RESPONSES_ENDPOINT: endpoint,
        DISTILL_PROMPT_VERSION: "qmd-expansion-teacher-v2-test",
        DISTILL_SYSTEM_PROMPT: "Preserve semantics.",
        DISTILL_USER_PROMPT_TEMPLATE: "Expand {{query}} neutrally.",
        DISTILL_MAX_OUTPUT_TOKENS: "1111",
      });

      const manifest = JSON.parse(await readFile(
        join(fixture.root, "runs/prompt-config-test/manifest.json"),
        "utf8",
      ));
      expect(manifest).toMatchObject({
        prompt_version: "qmd-expansion-teacher-v2-test",
        prompt_source: "environment",
        system_prompt: "Preserve semantics.",
        user_prompt_template: "Expand {{query}} neutrally.",
        max_output_tokens: 2222,
      });
      expect(manifest.prompt_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(bodies).toHaveLength(4);
      expect(bodies[0]).toMatchObject({
        max_output_tokens: 2222,
        input: [
          { role: "system", content: "Preserve semantics." },
          { role: "user", content: "Expand Claim alpha is not beta neutrally." },
        ],
      });
    });
  });

  test("derives a clean run and retries only failed seed slots", async () => {
    const fixture = await fixtureRoot();
    await withTeacherServer(async (endpoint, bodies) => {
      const seedDir = join(fixture.root, "runs/seed-v1");
      await mkdir(seedDir, { recursive: true });
      await writeFile(join(seedDir, "manifest.json"), `${JSON.stringify({
        version: "scifact-distill-run-v1",
        experiment_id: "seed-v1",
        split_sha256: fixture.splitSha256,
        teacher_provider: "openai-compatible",
        teacher_model: "gpt-test",
        prompt_version: "qmd-expansion-teacher-v1",
        api_endpoint: endpoint,
        reasoning_effort: "low",
        generated_queries: 1,
      })}\n`);
      const candidates = Array.from({ length: 4 }, (_, candidateIndex) => ({
        candidate_index: candidateIndex,
        generation_status: candidateIndex === 2 ? "generation_error" : "ok",
        raw_output: candidateIndex === 2 ? "" : `seed-output-${candidateIndex}`,
        parsed_output: candidateIndex === 2 ? [] : [["lex", `seed-${candidateIndex}`]],
        generation_error: candidateIndex === 2 ? "old failure" : null,
        contract: candidateIndex === 0 ? { valid: true } : null,
        metrics: candidateIndex === 0 ? { recall_at_30: 1, mrr_at_10: 1, ndcg_at_10: 1 } : null,
      }));
      await writeFile(join(seedDir, "candidates.jsonl"), `${JSON.stringify({
        qid: "q1",
        split: "train",
        query: "Claim alpha is not beta",
        raw_metrics: { recall_at_30: 1, mrr_at_10: 1, ndcg_at_10: 1 },
        candidates,
        selected_candidate_index: 0,
        selection_status: "winner",
      })}\n`);

      await runGenerator([
        "--experiment", "seed-v1-clean",
        "--model", "gpt-test",
        "--provider", "openai-compatible",
        "--root", fixture.root,
        "--queries", fixture.queriesPath,
        "--seed-run", seedDir,
        "--retry-generation-errors",
        "--max-output-tokens", "4096",
      ], {
        DISTILL_API_KEY: "test-key",
        DISTILL_RESPONSES_ENDPOINT: endpoint,
      });

      expect(bodies).toHaveLength(1);
      const output = JSON.parse((await readFile(
        join(fixture.root, "runs/seed-v1-clean/candidates.jsonl"),
        "utf8",
      )).trim());
      expect(output.candidates.map((candidate: { raw_output: string }) => candidate.raw_output))
        .toEqual([
          "seed-output-0",
          "seed-output-1",
          JSON.stringify({
            expansions: [
              { type: "lex", query: "alpha beta evidence" },
              { type: "vec", query: "evidence that alpha is not beta" },
            ],
          }),
          "seed-output-3",
        ]);
      expect(output.raw_metrics).toBeNull();
      expect(output.selected_candidate_index).toBeNull();
      expect(output.selection_status).toBe("pending");
      expect(output.candidates.every((candidate: { contract: unknown; metrics: unknown }) => (
        candidate.contract === null && candidate.metrics === null
      ))).toBe(true);

      const manifest = JSON.parse(await readFile(
        join(fixture.root, "runs/seed-v1-clean/manifest.json"),
        "utf8",
      ));
      expect(manifest).toMatchObject({
        seed_experiment_id: "seed-v1",
        seed_max_output_tokens: 1200,
        retry_seed_generation_errors: true,
        max_output_tokens: 4096,
        retry_generation_errors: {
          attempted_candidates: 1,
          recovered_candidates: 1,
          remaining_generation_errors: 0,
        },
      });
    });
  });
});
