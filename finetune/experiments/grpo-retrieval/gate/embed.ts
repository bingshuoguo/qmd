/**
 * Embed <dir>/embed-texts.json with the index's embedding model, using the same
 * query-side formatting qmd applies to vec/hyde lines at search time.
 * Writes <dir>/embed-vectors.f32 (row-major float32, same order as the texts).
 *
 *   node_modules/.bin/tsx finetune/experiments/grpo-retrieval/gate/embed.ts <dir>
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { formatQueryForEmbedding, getDefaultLlamaCpp } from "../../../../src/llm.js";

const dir = process.argv[2]!;
const texts = JSON.parse(readFileSync(join(dir, "embed-texts.json"), "utf8")) as string[];
const llm = getDefaultLlamaCpp();
const rows: number[][] = [];
const started = Date.now();
for (let i = 0; i < texts.length; i += 64) {
  const out = await llm.embedBatch(texts.slice(i, i + 64).map(t => formatQueryForEmbedding(t)));
  for (const [j, r] of out.entries()) {
    if (!r) throw new Error(`embedding failed for text ${i + j}`);
    rows.push(r.embedding);
  }
  if ((i / 64) % 20 === 0) process.stderr.write(`${rows.length}/${texts.length} ${((Date.now() - started) / 1000).toFixed(0)}s\n`);
}
const flat = new Float32Array(rows.length * rows[0]!.length);
rows.forEach((r, i) => flat.set(r, i * r.length));
writeFileSync(join(dir, "embed-vectors.f32"), Buffer.from(flat.buffer));
process.stderr.write(`done ${rows.length} x ${rows[0]!.length}\n`);
await llm.dispose();
process.exit(0);
