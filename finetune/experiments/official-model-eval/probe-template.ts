#!/usr/bin/env node
/**
 * Probe: how does node-llama-cpp render the official model's prompt, and does the
 * output change when we feed the exact training-format text instead?
 *   tsx finetune/experiments/official-model-eval/probe-template.ts <gguf> <queries.jsonl> [n]
 */
import { readFileSync } from "node:fs";
import { getLlama, LlamaChatSession, LlamaCompletion } from "node-llama-cpp";

const [modelPath, queriesPath, nArg] = process.argv.slice(2);
const n = Number(nArg ?? 4);
const queries = readFileSync(queriesPath!, "utf8").trim().split("\n").slice(0, n).map(l => JSON.parse(l).query as string);

const GRAMMAR = `
root ::= line+
line ::= type ": " content "\\n"
type ::= "lex" | "vec" | "hyde"
content ::= [^\\n]+
`;
const llama = await getLlama();
const model = await llama.loadModel({ modelPath: modelPath! });
const grammar = await llama.createGrammar({ grammar: GRAMMAR });

for (const q of queries) {
  const user = `/no_think Expand this search query: ${q}`;
  // (a) chat session, empty system prompt (what expand-official.ts does)
  const ctxA = await model.createContext({ contextSize: 2048 });
  const session = new LlamaChatSession({ contextSequence: ctxA.getSequence(), systemPrompt: "" });
  const rendered = session.chatWrapper.generateContextState({
    chatHistory: [{ type: "user", text: user }, { type: "model", response: [] }],
  }).contextText.toString();
  const wrapperName = session.chatWrapper.wrapperName;
  const a = await session.prompt(user, { grammar, temperature: 0, maxTokens: 400 });
  await ctxA.dispose();
  // (b) raw completion on the exact training text (no think block)
  const ctxB = await model.createContext({ contextSize: 2048 });
  const completion = new LlamaCompletion({ contextSequence: ctxB.getSequence() });
  const text = `<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n`;
  const b = await completion.generateCompletion(model.tokenize(text, true), { grammar, temperature: 0, maxTokens: 400 });
  await ctxB.dispose();
  console.log(`\n### ${q}\n-- wrapper=${wrapperName} rendered:\n${JSON.stringify(rendered)}`);
  console.log(`-- (a) chat session:\n${a}\n-- (b) raw training-format completion:\n${b}`);
}
await model.dispose();
