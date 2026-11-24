import { readFileSync } from "node:fs";
import { getLlama, LlamaCompletion } from "node-llama-cpp";
const [modelPath, queriesPath] = process.argv.slice(2);
const qs = readFileSync(queriesPath!, "utf8").trim().split("\n").slice(0, 3).map(l => JSON.parse(l).query);
qs.push("auth config", "ancient Greece democracy Athens");
const llama = await getLlama(); const model = await llama.loadModel({ modelPath: modelPath! });
for (const q of qs) {
  const ctx = await model.createContext({ contextSize: 2048 });
  const c = new LlamaCompletion({ contextSequence: ctx.getSequence() });
  const text = `<|im_start|>user\n/no_think Expand this search query: ${q}<|im_end|>\n<|im_start|>assistant\n`;
  const out = await c.generateCompletion(model.tokenize(text, true), { temperature: 0, maxTokens: 300 });
  console.log(`\n### ${q}\n${JSON.stringify(out)}`);
  await ctx.dispose();
}
