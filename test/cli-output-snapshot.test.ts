/**
 * Golden-file output snapshots for CLI commands.
 *
 * Guards the qmd.ts file split (docs/specs/qmd-cli-file-split.md): recorded
 * against the monolithic qmd.ts, and must match byte-for-byte after every
 * move batch. Environment-volatile fields (temp paths, relative times, index
 * file size) are normalized before comparison; everything else — including
 * stderr text and exit codes — is compared exactly.
 *
 * Re-record baselines after intentional output changes:
 *   QMD_UPDATE_SNAPSHOTS=1 npx vitest run test/cli-output-snapshot.test.ts
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, mkdir, writeFile, utimes, readFile, rm, readdir } from "fs/promises";
import { existsSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { realpathSync } from "fs";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const snapshotDir = join(thisDir, "_snapshots", "cli-output");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const qmdCommand = isBunRuntime
  ? { command: process.execPath, args: [qmdScript] }
  : { command: process.execPath, args: [tsxCli, qmdScript] };

const UPDATE = process.env.QMD_UPDATE_SNAPSHOTS === "1";

let testDir: string;
let testDbPath: string;
let testConfigDir: string;
let testCacheDir: string;
let fixturesDir: string;
let realTestDir: string;

// Fixed mtime for fixture files so `ls` time columns are deterministic.
const FIXTURE_MTIME = new Date("2024-01-15T12:00:00Z");

async function runQmd(
  args: string[],
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const proc = spawn(qmdCommand.command, [...qmdCommand.args, ...args], {
    cwd: fixturesDir,
    env: {
      ...process.env,
      INDEX_PATH: testDbPath,
      QMD_CONFIG_DIR: testConfigDir,
      XDG_CACHE_HOME: testCacheDir,
      PWD: fixturesDir, // getPwd() checks this
      TZ: "UTC", // formatLsTime renders fixture mtimes in local time
      QMD_DOCTOR_DEVICE_PROBE: "0",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  const stdoutPromise = new Promise<string>((resolve, reject) => {
    let data = "";
    proc.stdout?.on("data", (chunk: Buffer) => { data += chunk.toString(); });
    proc.once("error", reject);
    proc.stdout?.once("end", () => resolve(data));
  });
  const stderrPromise = new Promise<string>((resolve, reject) => {
    let data = "";
    proc.stderr?.on("data", (chunk: Buffer) => { data += chunk.toString(); });
    proc.once("error", reject);
    proc.stderr?.once("end", () => resolve(data));
  });
  const exitCode = await new Promise<number>((resolve, reject) => {
    proc.once("error", reject);
    proc.on("close", (code) => resolve(code ?? 1));
  });

  return { stdout: await stdoutPromise, stderr: await stderrPromise, exitCode };
}

// Mask the four volatile classes: machine-specific paths, relative times
// ("3m ago"), the SQLite index byte size, and the git commit hash embedded
// in `qmd --version` output.
function normalize(s: string): string {
  return s
    .split(realTestDir).join("<TMP>")
    .split(testDir).join("<TMP>")
    .replace(/\d+[smhd] ago/g, "<AGO>")
    .replace(/^Size:.*$/gm, "Size: <SIZE>")
    .replace(/\([0-9a-f]{7,40}\)/g, "(<COMMIT>)");
}

function goldenBody(result: { stdout: string; stderr: string; exitCode: number }): string {
  return `exit: ${result.exitCode}\n--- stderr ---\n${normalize(result.stderr)}--- stdout ---\n${normalize(result.stdout)}`;
}

const CASES: { name: string; args: string[] }[] = [
  // search × 6 formats
  { name: "search-cli", args: ["search", "meeting", "-n", "3"] },
  { name: "search-json", args: ["search", "meeting", "-n", "3", "--json"] },
  { name: "search-csv", args: ["search", "meeting", "-n", "3", "--csv"] },
  { name: "search-md", args: ["search", "meeting", "-n", "3", "--md"] },
  { name: "search-xml", args: ["search", "meeting", "-n", "3", "--xml"] },
  { name: "search-files", args: ["search", "meeting", "-n", "3", "--files"] },
  // empty-result search (printEmptySearchResults) × 6 formats
  { name: "search-empty-cli", args: ["search", "zzqxxnomatch", "-n", "3"] },
  { name: "search-empty-json", args: ["search", "zzqxxnomatch", "-n", "3", "--json"] },
  { name: "search-empty-csv", args: ["search", "zzqxxnomatch", "-n", "3", "--csv"] },
  { name: "search-empty-md", args: ["search", "zzqxxnomatch", "-n", "3", "--md"] },
  { name: "search-empty-xml", args: ["search", "zzqxxnomatch", "-n", "3", "--xml"] },
  { name: "search-empty-files", args: ["search", "zzqxxnomatch", "-n", "3", "--files"] },
  // min-score filtered empty (the other EmptySearchReason)
  { name: "search-minscore-cli", args: ["search", "meeting", "-n", "3", "--min-score", "99"] },
  // multi-get × 6 formats
  { name: "multiget-cli", args: ["multi-get", "notes/*.md"] },
  { name: "multiget-json", args: ["multi-get", "notes/*.md", "--json"] },
  { name: "multiget-csv", args: ["multi-get", "notes/*.md", "--csv"] },
  { name: "multiget-md", args: ["multi-get", "notes/*.md", "--md"] },
  { name: "multiget-xml", args: ["multi-get", "notes/*.md", "--xml"] },
  { name: "multiget-files", args: ["multi-get", "notes/*.md", "--files"] },
  // get / ls / status
  { name: "get-cli", args: ["get", "qmd://snap/notes/meeting.md"] },
  { name: "get-range", args: ["get", "qmd://snap/README.md:2:3"] },
  { name: "ls-collections", args: ["ls"] },
  { name: "ls-collection", args: ["ls", "snap"] },
  { name: "status", args: ["status"] },
  // help / version (help.ts move coverage)
  { name: "help", args: ["--help"] },
  { name: "help-noargs", args: [] },
  { name: "version", args: ["--version"] },
  { name: "skill-help", args: ["skill", "-h"] },
  { name: "skill-help-subcmd", args: ["skill", "help"] },
  { name: "collection-help", args: ["collection", "help"] },
  { name: "context-noargs", args: ["context"] },
  // error paths: stderr text + exit codes
  { name: "error-get-missing", args: ["get", "qmd://snap/nope.md"] },
  { name: "error-unknown-command", args: ["frobnicate"] },
  { name: "error-search-noquery", args: ["search"] },
];

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-snapshot-"));
  realTestDir = realpathSync(testDir); // macOS /var → /private/var
  testDbPath = join(testDir, "index.sqlite");
  testConfigDir = join(testDir, "config");
  testCacheDir = join(testDir, "cache");
  fixturesDir = join(testDir, "fixtures");

  await mkdir(testConfigDir, { recursive: true });
  await mkdir(testCacheDir, { recursive: true });
  await mkdir(join(fixturesDir, "notes"), { recursive: true });
  await writeFile(join(testConfigDir, "index.yml"), "collections: {}\n");

  await writeFile(
    join(fixturesDir, "README.md"),
    `# Snapshot Fixture

Root document for the snapshot collection.

## Scope

- Deterministic content for golden-file output comparison
- Never edited without re-recording snapshots
`
  );
  await writeFile(
    join(fixturesDir, "notes", "meeting.md"),
    `# Weekly Sync Meeting

Date: 2024-01-15

## Attendees
- Alice
- Bob

## Notes
- Reviewed the meeting agenda
- Bob owns the follow-up
`
  );
  await writeFile(
    join(fixturesDir, "notes", "ideas.md"),
    `# Idea Parking Lot

## Ideas
- Keyboard-first navigation
- Export snapshots to PDF
`
  );

  // Pin fixture mtimes so `ls` renders a fixed date column.
  for (const rel of ["README.md", join("notes", "meeting.md"), join("notes", "ideas.md")]) {
    await utimes(join(fixturesDir, rel), FIXTURE_MTIME, FIXTURE_MTIME);
  }

  const add = await runQmd(["collection", "add", fixturesDir, "--name", "snap"]);
  if (add.exitCode !== 0) throw new Error(`collection add failed: ${add.stderr}`);
  const ctx = await runQmd(["context", "add", "qmd://snap/", "Snapshot fixture collection"]);
  if (ctx.exitCode !== 0) throw new Error(`context add failed: ${ctx.stderr}`);

  if (UPDATE) await mkdir(snapshotDir, { recursive: true });
});

afterAll(async () => {
  if (testDir) await rm(testDir, { recursive: true, force: true });
});

describe("CLI output snapshots", () => {
  for (const c of CASES) {
    test(c.name, async () => {
      const result = await runQmd(c.args);
      const body = goldenBody(result);
      const goldenPath = join(snapshotDir, `${c.name}.golden`);

      if (UPDATE) {
        const { writeFile: write } = await import("fs/promises");
        await write(goldenPath, body);
        return;
      }

      expect(existsSync(goldenPath), `missing golden ${c.name}.golden — record with QMD_UPDATE_SNAPSHOTS=1`).toBe(true);
      const expected = await readFile(goldenPath, "utf8");
      expect(body, `snapshot mismatch for ${c.name}`).toBe(expected);
    });
  }

  test("no stale golden files", async () => {
    if (UPDATE || !existsSync(snapshotDir)) return;
    const onDisk = (await readdir(snapshotDir)).filter((f) => f.endsWith(".golden")).sort();
    const expected = CASES.map((c) => `${c.name}.golden`).sort();
    expect(onDisk).toEqual(expected);
  });
});
