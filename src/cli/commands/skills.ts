/**
 * commands/skills.ts - bundled runtime skill discovery, `qmd skills
 * list/get/path`, and `qmd skill show/install` (including the Claude symlink
 * dance). Self-contained: no store/db access; reads skill files from disk.
 */
import { readFileSync, readdirSync, realpathSync, statSync, existsSync, unlinkSync, writeFileSync, mkdirSync, lstatSync, rmSync, symlinkSync, readlinkSync, copyFileSync } from "fs";
import { basename, dirname, relative as relativePath } from "path";
import { fileURLToPath } from "url";
import { createInterface } from "readline/promises";
import { getPwd, homedir, resolve } from "../../store.js";
import { showSkillsHelp } from "../help.js";

function getSkillInstallDir(globalInstall: boolean): string {
  return globalInstall
    ? resolve(homedir(), ".agents", "skills", "qmd")
    : resolve(getPwd(), ".agents", "skills", "qmd");
}

function getClaudeSkillLinkPath(globalInstall: boolean): string {
  return globalInstall
    ? resolve(homedir(), ".claude", "skills", "qmd")
    : resolve(getPwd(), ".claude", "skills", "qmd");
}

function pathExists(path: string): boolean {
  try {
    lstatSync(path);
    return true;
  } catch {
    return false;
  }
}

function removePath(path: string): void {
  const stat = lstatSync(path);
  if (stat.isDirectory() && !stat.isSymbolicLink()) {
    rmSync(path, { recursive: true, force: true });
  } else {
    unlinkSync(path);
  }
}

type SkillInfo = {
  name: string;
  description: string;
  dir: string;
  hidden: boolean;
};

const SKILL_DIR = "skills";

function findPackageRoot(): string | null {
  if (process.env.QMD_SKILLS_DIR) {
    return null;
  }

  const start = dirname(fileURLToPath(import.meta.url));
  let current = start;
  while (true) {
    if (existsSync(resolve(current, SKILL_DIR))) {
      return current;
    }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return null;
}

function getSkillSearchDirs(_runtimeOnly = false): string[] {
  if (process.env.QMD_SKILLS_DIR) {
    return [process.env.QMD_SKILLS_DIR];
  }

  const root = findPackageRoot();
  if (!root) return [];

  const dir = resolve(root, SKILL_DIR);
  return existsSync(dir) ? [dir] : [];
}

function parseSkillFrontmatter(content: string): { name: string; description: string; hidden: boolean } | null {
  const trimmed = content.trimStart();
  if (!trimmed.startsWith("---")) return null;
  const end = trimmed.slice(3).indexOf("\n---");
  if (end < 0) return null;

  const frontmatter = trimmed.slice(3, 3 + end);
  let name = "";
  let description = "";
  let hidden = false;
  const lines = frontmatter.split(/\r?\n/);

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    if (line.startsWith("name:")) {
      name = line.slice("name:".length).trim();
    } else if (line.startsWith("description:")) {
      const parts = [line.slice("description:".length).trim()];
      while (i + 1 < lines.length && /^\s+\S/.test(lines[i + 1]!)) {
        i++;
        parts.push(lines[i]!.trim());
      }
      description = parts.join(" ");
    } else if (line.startsWith("hidden:")) {
      const value = line.slice("hidden:".length).trim().toLowerCase();
      hidden = value === "true" || value === "yes";
    }
  }

  if (!name) return null;
  return { name, description, hidden };
}

function discoverSkills(runtimeOnly = false): SkillInfo[] {
  const skills: SkillInfo[] = [];
  for (const dir of getSkillSearchDirs(runtimeOnly)) {
    let entries: string[] = [];
    try {
      entries = readdirSync(dir);
    } catch {
      continue;
    }

    for (const entry of entries) {
      const skillDir = resolve(dir, entry);
      const skillPath = resolve(skillDir, "SKILL.md");
      if (!existsSync(skillPath)) continue;
      let content = "";
      try {
        content = readFileSync(skillPath, "utf-8");
      } catch {
        continue;
      }
      const parsed = parseSkillFrontmatter(content);
      if (!parsed) continue;
      skills.push({ ...parsed, dir: skillDir });
    }
  }
  return skills.sort((a, b) => a.name.localeCompare(b.name));
}

function findSkill(name: string, runtimeOnly = false): SkillInfo | null {
  return discoverSkills(runtimeOnly).find((skill) => skill.name === name) ?? null;
}

function readSkillContent(skill: SkillInfo): string {
  return readFileSync(resolve(skill.dir, "SKILL.md"), "utf-8");
}

function collectSkillFiles(skill: SkillInfo): { relativePath: string; content: string }[] {
  const files: { relativePath: string; content: string }[] = [];
  for (const subdirName of ["references", "templates", "scripts"]) {
    const subdir = resolve(skill.dir, subdirName);
    if (!existsSync(subdir)) continue;
    for (const entry of readdirSync(subdir).sort()) {
      const filePath = resolve(subdir, entry);
      try {
        if (!statSync(filePath).isFile()) continue;
        files.push({ relativePath: `${subdirName}/${basename(filePath)}`, content: readFileSync(filePath, "utf-8") });
      } catch {
        // Ignore unreadable supplementary files.
      }
    }
  }
  return files;
}

function showSkill(): void {
  const skill = findSkill("qmd");
  if (!skill) {
    throw new Error("QMD skill not found. Reinstall qmd or set QMD_SKILLS_DIR.");
  }
  console.log("QMD Skill");
  console.log("");
  const content = readSkillContent(skill);
  process.stdout.write(content.endsWith("\n") ? content : content + "\n");
}

function copyDirectoryContents(sourceDir: string, targetDir: string): void {
  mkdirSync(targetDir, { recursive: true });
  for (const entry of readdirSync(sourceDir)) {
    const sourcePath = resolve(sourceDir, entry);
    const targetPath = resolve(targetDir, entry);
    const stat = statSync(sourcePath);
    if (stat.isDirectory()) {
      copyDirectoryContents(sourcePath, targetPath);
    } else if (stat.isFile()) {
      copyFileSync(sourcePath, targetPath);
    }
  }
}

function installedSkillStubContent(): string {
  return `---
name: qmd
description: Bootstrap QMD search instructions from the installed qmd CLI. Use when users ask to find notes, retrieve documents, inspect a wiki, or answer from indexed local markdown.
license: MIT
compatibility: Requires qmd CLI. Run \`qmd skill show\` for version-matched instructions.
allowed-tools: Bash(qmd:*), mcp__qmd__*
---

# QMD - Query Markdown Documents

This installed skill is intentionally a small bootstrap so it does not go stale
when the qmd package updates.

Load the full, version-matched QMD instructions from the CLI:

!\`qmd skill show\`

If your agent does not support bang-command expansion, run:

\`\`\`bash
qmd skill show
\`\`\`

Then follow those instructions. In short: search first, fetch full sources with
\`qmd get\` or \`qmd multi-get\`, and answer from retrieved text rather than snippets.
`;
}

function writeSkillInstall(targetDir: string, force: boolean): void {
  if (pathExists(targetDir)) {
    if (!force) {
      throw new Error(`Skill already exists: ${targetDir} (use --force to replace it)`);
    }
    removePath(targetDir);
  }

  const skill = findSkill("qmd");
  if (!skill) {
    throw new Error("QMD skill not found. Reinstall qmd or set QMD_SKILLS_DIR.");
  }

  copyDirectoryContents(skill.dir, targetDir);
  writeFileSync(resolve(targetDir, "SKILL.md"), installedSkillStubContent(), "utf-8");
}

function outputSkillsJson(payload: unknown): void {
  console.log(JSON.stringify(payload));
}

function runSkillsCommand(args: string[], jsonMode: boolean, fullOption = false, allOption = false): void {
  const subcommand = args[0] ?? "list";
  const runtimeSkills = () => discoverSkills(true).filter((skill) => !skill.hidden);

  switch (subcommand) {
    case "list": {
      const skills = runtimeSkills();
      if (jsonMode) {
        outputSkillsJson({ success: true, data: skills.map(({ name, description }) => ({ name, description })) });
        return;
      }
      if (skills.length === 0) {
        console.log("No skills found");
        return;
      }
      const maxName = Math.max(...skills.map((skill) => skill.name.length));
      for (const skill of skills) {
        console.log(`  ${skill.name.padEnd(maxName)}  ${skill.description}`);
      }
      return;
    }

    case "get": {
      const full = fullOption || args.includes("--full");
      const getAll = allOption || args.includes("--all");
      const names = args.slice(1).filter((arg) => arg !== "--full" && arg !== "--all");
      const targets = getAll ? runtimeSkills() : names.map((name) => {
        const skill = findSkill(name, true);
        if (!skill) {
          throw new Error(`Skill not found: ${name}`);
        }
        return skill;
      });

      if (targets.length === 0) {
        throw new Error("No skill name provided. Usage: qmd skills get <name>");
      }

      if (jsonMode) {
        outputSkillsJson({
          success: true,
          data: targets.map((skill) => ({
            name: skill.name,
            content: readSkillContent(skill),
            ...(full ? { files: collectSkillFiles(skill).map((file) => ({ path: file.relativePath, content: file.content })) } : {}),
          })),
        });
        return;
      }

      targets.forEach((skill, index) => {
        if (index > 0) console.log("\n---\n");
        const content = readSkillContent(skill);
        process.stdout.write(content.endsWith("\n") ? content : content + "\n");
        if (full) {
          for (const file of collectSkillFiles(skill)) {
            console.log(`\n--- ${file.relativePath} ---\n`);
            process.stdout.write(file.content.endsWith("\n") ? file.content : file.content + "\n");
          }
        }
      });
      return;
    }

    case "path": {
      const name = args[1];
      if (!name) {
        const paths = getSkillSearchDirs(true);
        if (jsonMode) outputSkillsJson({ success: true, data: { paths } });
        else paths.forEach((path) => console.log(path));
        return;
      }
      const skill = findSkill(name, true);
      if (!skill) {
        throw new Error(`Skill not found: ${name}`);
      }
      if (jsonMode) outputSkillsJson({ success: true, data: { name: skill.name, path: skill.dir } });
      else console.log(skill.dir);
      return;
    }

    case "help": {
      showSkillsHelp();
      return;
    }

    default:
      throw new Error(`Unknown skills subcommand: ${subcommand}`);
  }
}

function ensureClaudeSymlink(linkPath: string, targetDir: string, force: boolean): boolean {
  const parentDir = dirname(linkPath);
  if (pathExists(parentDir)) {
    const resolvedTargetDir = realpathSync(dirname(targetDir));
    const resolvedLinkParent = realpathSync(parentDir);

    // If .claude/skills already resolves to the same directory as .agents/skills,
    // the skill is already visible to Claude and creating qmd -> qmd would loop.
    if (resolvedTargetDir === resolvedLinkParent) {
      return false;
    }
  }

  const linkTarget = relativePath(parentDir, targetDir) || ".";

  mkdirSync(parentDir, { recursive: true });

  if (pathExists(linkPath)) {
    const stat = lstatSync(linkPath);
    if (stat.isSymbolicLink() && readlinkSync(linkPath) === linkTarget) {
      return true;
    }
    if (!force) {
      throw new Error(`Claude skill path already exists: ${linkPath} (use --force to replace it)`);
    }
    removePath(linkPath);
  }

  symlinkSync(linkTarget, linkPath, "dir");
  return true;
}

async function shouldCreateClaudeSymlink(linkPath: string, autoYes: boolean): Promise<boolean> {
  if (autoYes) {
    return true;
  }
  if (!process.stdin.isTTY || !process.stdout.isTTY) {
    console.log(`Tip: create a Claude symlink manually at ${linkPath}`);
    return false;
  }

  const rl = createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  try {
    const answer = await rl.question(`Create a symlink in ${linkPath}? [y/N] `);
    const normalized = answer.trim().toLowerCase();
    return normalized === "y" || normalized === "yes";
  } finally {
    rl.close();
  }
}

async function installSkill(globalInstall: boolean, force: boolean, autoYes: boolean): Promise<void> {
  const installDir = getSkillInstallDir(globalInstall);
  writeSkillInstall(installDir, force);
  console.log(`✓ Installed QMD skill to ${installDir}`);

  const claudeLinkPath = getClaudeSkillLinkPath(globalInstall);
  if (!(await shouldCreateClaudeSymlink(claudeLinkPath, autoYes))) {
    return;
  }

  const linked = ensureClaudeSymlink(claudeLinkPath, installDir, force);
  if (linked) {
    console.log(`✓ Linked Claude skill at ${claudeLinkPath}`);
  } else {
    console.log(`✓ Claude already sees the skill via ${dirname(claudeLinkPath)}`);
  }
}

export {
  runSkillsCommand,
  showSkill,
  installSkill,
  outputSkillsJson,
};
