# qmd.ts SQL 所有权重构 Spec

| 项 | 值 |
|---|---|
| 状态 | 评审中(v2 — P0/P1 已修正,待二次确认) |
| 日期 | 2026-08-02(v1)→ 2026-08-02(v2) |
| 范围 | **仅 Phase 1**(SQL 下沉);Phase 3 / Phase 2 仅作路线图,不承诺 |
| 涉及文件 | `src/cli/qmd.ts`、`src/store.ts` |
| 铁律 | 不改任何用户可见行为(纯代码搬移) |

> **v2 修订**:接受评审意见,删除 v1 §8.1 的 `getStatus` 复用方案(违反 D4 逐字搬移),全部改为"一条 SQL 对应一个窄函数";修正字段名错误;补充 `findDocuments` 既有重复记录;验证补类型/构建检查。

---

## 1. 背景与真实问题

CLI 层 `qmd.ts` 绕过数据层 `store.ts` 直接 `db.prepare(SQL)`。这不是"分层洁癖",而是**同一份 schema 知识写了两遍**:

- 全项目 `CREATE TABLE` **只出现在 `src/store.ts`**(6 处),schema 是它独有的知识。
- schema **会持续演进**:`store.ts` 内有 `PRAGMA user_version` 迁移、`ALTER TABLE content_vectors ADD COLUMN embed_fingerprint`、legacy 表清理。
- `qmd.ts` 里 **18 处** `db.prepare` 全部打在这些 store 拥有的表上(`documents`/`content`/`content_vectors`/`store_config`/`vectors_vec`/`sqlite_master`)。

**失败模式**:下次改 schema 的人会改 `store.ts`、测试全绿,却漏掉 `qmd.ts` 里那几处内联 SQL——SQL 过期通常**不崩,只静默查错数据**(典型如 doctor 的 fingerprint 分组)。

**既有重复(实锤)**:`store.ts` 的 [`findDocuments`](../../src/store.ts) 已实现 multiGet 的文件解析逻辑(逗号/glob 判断条件与 qmd.ts 逐字相同),qmd.ts 又用裸 SQL 内联了一遍。**已是 1.5 套实现**,本 spec 不得再造出第三套(见 §8.3)。

> 定性:这是**预防性 + 去重**,不是修 bug。今天没有故障。价值成立的依据是这库有改 schema 的前科。

---

## 2. 核心数据与所有权

| 数据 | 拥有者 | 现状 |
|---|---|---|
| collections / contexts / models | `collections.js`(YAML) | 已走接口 ✅ |
| documents / content / content_vectors / vectors_vec / store_config | `store.ts`(SQLite) | **被 qmd.ts 越权** ❌ ← 本 spec |
| embed / generate / rerank 推理 | `llm.js` | 已走接口 ✅ |

---

## 3. 决策记录(评审锁定)

| # | 决策 | 结论 | 理由 |
|---|---|---|---|
| D1 | 范围 | **只做 Phase 1**;P3(拆文件)/P2(统一生命周期)写成路线图 | 最小改动解决真问题;每批可验证、随时可停 |
| D2 | 边界 | **绝对 I1**——qmd.ts 内联 SQL 全下沉,结束 `grep db.prepare src/cli/qmd.ts` 为 0 | 带例外的不变式不是不变式 |
| D3 | 落点 | 沉入 **`store.ts`**,挨着 `listCollections`/`findDocument` 等同类;不新建模块 | 一致性优先;store.ts 的拆分另立项 |
| D4 | 粒度 | **一条 SQL 对应一个窄函数,逐字搬移**。不合并 multiGet 的 3 个查询,不聚 versions,不拼 aggregates | 重构不改行为;粒度=原 SQL 粒度,最不易引入语义漂移 |
| D5 | 不复用 | **Phase 1 不复用 `getStatus` / `findDocuments`**。它们做的事比原窄 SQL 多(聚合、needsEmbedding),复用=改行为 | 评审 P1:复用违反 D4。窄函数逐字搬才零语义变化 |
| D6 | 内部性 | 新函数作为 store.ts **模块级 export**,**不加入 `src/index.ts` 公共 API,不挂到 store 对象返回的 interface** | 避免把一次 CLI 重构扩大成永久 SDK 表面 |

---

## 4. 不变式

- **I1** — `qmd.ts` 不出现任何 `db.prepare` / SQL 字符串;所有库读写经 `store.ts` 函数。
- **I2** — 各子命令 stdout/stderr 文本、6 种输出格式、退出码**逐字节不变**。
- **I3** — `finishSuccessfulCliCommand` 的 `flush → cleanup → exitCode=0`(不调 `process.exit(0)`)语义保留,darwin Metal teardown 不回退。
- **I4** — 测试依赖的导出符号(`buildEditorUri`/`termLink`/`finishSuccessfulCliCommand`/`resolve*ForCli`)保持可从 `qmd.ts` 导入。

---

## 5. 用户可见行为契约

CLI 输出即 userspace,`Never break userspace` 适用。本重构**零行为变化**:不改一个字符的输出、不改任何退出码、不改 SQL 语义(只搬不写)。

特别保留的既有行为:
- `listFiles` **先遍历 YAML collection**,再逐个查计数 → **零文档 collection 也显示 `0 files`**(不得改用"从 documents GROUP BY"的方式,否则零文档 collection 消失)。
- doctor 的 `sqlite_version()` 与 `vec_version()` 是**两个独立 try/catch**,一个失败不影响另一个的结果展示。

---

## 6. 非目标(明确拒绝)

- ❌ 不改输出格式/文案/颜色。
- ❌ 不重写 SQL 语义、不合并 multiGet 的 3 个查询。
- ❌ 不复用 `getStatus` / `findDocuments`(见 D5;收敛另立项)。
- ❌ 不动 `store.ts` 的数据逻辑(只新增窄查询函数 / 收编 store 内部语句)。
- ❌ 不拆 `store.ts`、不拆 `qmd.ts`(那是 Phase 3)。
- ❌ 不碰 `parseVirtualPath` 路径解析特殊案例(独立问题,另立项)。
- ❌ 不碰 Metal workaround、不清理与本次移动无关的既有代码。

---

## 7. 破坏风险(按概率)

1. **输出漂移** — 最高风险;一处文案/换行改动即破 userspace 与测试。
2. **语义漂移** — 搬 SQL 时手滑改了 `WHERE`/`JOIN`/`LIMIT`/`GROUP BY`。
3. **行为等价缺口** — 把"多条独立语句"合并成一个函数,或改用更高层聚合函数(getStatus),会改变计算量与失败语义(v1 在此栽过)。
4. **`findDocuments` 第三套实现** — 若新窄函数与 `findDocuments` 语义重叠却不记录,会形成长期重复。

---

## 8. Phase 1 实施方案

原则:**逐字、按原 SQL 粒度、一个窄函数对应一条语句**(D4/D5)。不复用 `getStatus`/`findDocuments`。

### 8.1 新增 store.ts 窄查询函数(逐字搬入 qmd.ts 现有 SQL)

| 新函数(建议名) | qmd.ts 站点 | 逐字保留的 SQL |
|---|---|---|
| `invalidateConfigCache(db)` | `:158` | `DELETE FROM store_config WHERE key='config_hash'` |
| `countActiveDocuments(db)` | `:484` `:3627` | `SELECT COUNT(*) FROM documents WHERE active=1` |
| `countContentVectors(db)` | `:485` | `SELECT COUNT(*) FROM content_vectors` |
| `getLatestDocumentModifiedAt(db)` | `:490` | `SELECT MAX(modified_at) FROM documents WHERE active=1` |
| `findDocumentRef(db, query)` | `:1090` `:1103` `:1117` | qmd://精确、path 精确、path 后缀 三分支解析(逐字)→ `{collection,path,hash,bodyLength}` |
| `getDocumentHash(db, collection, path)` | `:1177` | `SELECT d.hash FROM documents WHERE collection=? AND path=? AND active=1` |
| `getDocumentContent(db, collection, path)` | `:1212` | `SELECT content.doc, d.title ... WHERE collection=? AND path=? AND active=1` |
| `countDocumentsInCollection(db, name)` | `:1363` | `SELECT COUNT(*) FROM documents WHERE collection=? AND active=1` |
| `listDocumentsWithMeta(db, collection, pathPrefix?)` | `:1480` | 列出 `{path,title,modified_at,size}`(含 pathPrefix 两变体) |
| `hasVectorTable(db)` | `:3632` | `SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_vec'` |
| `sampleEmbeddedChunks(db, model, fingerprint, n)` | `:3637` | 向量采样 SELECT(content_vectors JOIN documents JOIN content) |
| `getStoredEmbedding(db, hashSeq)` | `:3672` | `SELECT embedding FROM vectors_vec WHERE hash_seq=?` |
| `getSqliteVersion(db)` | `:3844` | `SELECT sqlite_version()`(**独立函数**) |
| `getVecVersion(db)` | `:3854` | `SELECT vec_version()`(**独立函数**,不与上合并) |
| `getEmbeddingFingerprintGroups(db)` | `:3888` | fingerprint `GROUP BY`(doctor) |

15 个窄函数覆盖全部 18 处(`countActiveDocuments` 复用于 2 站、`findDocumentRef` 收编 3 站)。

### 8.2 关键行为等价约束

- **`listFiles`**:`countDocumentsInCollection` 只替换 `:1363` 的计数 SQL;**YAML 遍历逻辑原样留在 qmd.ts**,保证零文档 collection 仍显示 `0 files`。
- **`getSqliteVersion` / `getVecVersion`**:两个独立函数,qmd.ts 调用点**各自保留原有 try/catch**,任一失败不影响另一个。
- **`showStatus`**:仍自行调用已有的 `getHashesNeedingEmbedding`(:487),不引入 `getStatus`。

### 8.3 既有重复警示(必须记录,不在 Phase 1 处理)

`store.ts` 的 [`findDocuments(db, pattern, {includeBody, maxBytes})`](../../src/store.ts)(:4338)已实现 multiGet 的"逗号/glob 解析 + 取 body",与 qmd.ts `multiGet` 的内联 SQL 高度重叠。**Phase 1 为保行为等价,不把 multiGet 切到 `findDocuments`**(它还缺 context/docid/full-path/maxLines/格式化)。**后续单独立项收敛二者**,避免长期并存第三套实现。本 Phase 新增的 `findDocumentRef`/`getDocumentContent` 与 `findDocuments` 的语义差异须在其 docstring 中注明。

### 8.4 内部 interface

新函数仅作 `store.ts` 模块级 export;`src/index.ts` 的公共导出**不新增**,store 对象返回的 interface 也**不挂载**——防止 CLI 重构泄漏成 SDK 表面。

---

## 9. 验证

现有测试网兜底大部分契约,**逐阶段跑**:

1. **类型与构建**(v2 新增,最先跑):
   - `npm run test:types`(tsc --noEmit)—— 捕获字段名/返回类型错误(正是 v1 P0 那类)。
   - `npm run build`。
2. `npx vitest run test/` 全绿(重构前先确认基线绿)。
   - `test/cli.test.ts`:起真实子进程,逐命令断言 stdout + 退出码。
   - `test/cli-exit-lifecycle.test.ts`:锁 Metal 生命周期(I3)。
   - `test/path-fidelity.test.ts`:锁路径展示。
3. **输出 diff 抽查**(固定 fixture + 固定环境 + 固定 DB snapshot):
   - 用固定小语料 collection + 同一份 `index.sqlite` 快照,重构前后各跑一次。
   - `NO_COLOR=1`、固定 `QMD_EDITOR_URI`、固定环境变量后 diff。
   - **确定性命令**(`ls`/`multi-get`/`get`)逐字节 diff 应为空。
   - **非确定命令**(`status` 的相对时间、`doctor` 的设备探针/env overrides)不做全文 diff,改为比对结构 + 关键行,或在同一时间窗/同一环境下抽查。
4. **退出码抽查**:`qmd get <不存在>` → 1;`qmd collection remove <不存在>` → 1;成功路径 → 0。

---

## 10. 路线图(仅记录,不在本次承诺内)

- **Phase 3 — 拆分 god-file**:前提是 Phase 1 已完成。拆出 `doctor.ts`/`skills.ts`/`output.ts`,`qmd.ts` 只留 `parseCLI` + 分发。纯搬移。
- **Phase 2 — 统一命令生命周期**:单个 command runner 收编 38 处 `closeDb` + 69 处 `process.exit`,严格保留 Metal 路径。风险最高,最后做、可缓。
- **另行立项**:`parseVirtualPath` 路径解析特殊案例;`store.ts` 自身拆分(5471 行);**`multiGet` 内联实现与 `findDocuments` 的收敛**(见 §8.3)。

---

## 11. 成功标准

```
0. 类型/构建  → verify: npm run test:types 通过;npm run build 通过
1. SQL 下沉   → verify: grep -cE "db\.prepare|db\.exec" src/cli/qmd.ts == 0
2. 行为不变   → verify: 固定环境下 ls/multi-get/get 输出 diff 为空;status/doctor 结构一致;退出码抽查一致
3. 测试契约   → verify: test/cli.test.ts、cli-exit-lifecycle、path-fidelity 全绿
```
