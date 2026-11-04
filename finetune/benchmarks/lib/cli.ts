/**
 * Shared helpers for the benchmark CLI entrypoints. Importing this module is
 * side-effect free: applyLlamaEnvMitigation() mutates process.env and is only
 * ever called from a CLI body, never at module top level.
 */

import { renameSync, writeFileSync } from "node:fs";
import { platform } from "node:os";

/** Atomic JSON write: temp file in the same directory, then rename. */
export function writeJson(path: string, value: unknown): void {
  const temporaryPath = `${path}.tmp-${process.pid}`;
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, path);
}

/**
 * Match the qmd launcher mitigation before node-llama-cpp loads on Apple
 * Silicon. Never overrides an explicit user setting.
 */
export function applyLlamaEnvMitigation(): void {
  if (platform() === "darwin" && !process.env.GGML_METAL_NO_RESIDENCY) {
    process.env.GGML_METAL_NO_RESIDENCY = "1";
  }
}
