# workflow-RAG 方案

本文是 **workflow-RAG** 这条管线的规范，与 [data-model.md](data-model.md) 同体例。data-model 描述现役管线
（`A → B → C`，本文按新名写作 `perceiver → writer → reviewer`）；本文描述与它并列的第二条管线：

> **perceiver → writer → mimic（检索 + 改写）→ reviewer**

以及它依赖的现代诗检索资产（`sheepzh/poetry` + bge-large-zh-v1.5 + Chroma）。

**状态：设计稿，代码尚未落地。** 落地要改哪些文件见第 8 节。

与 data-model 有一处表面冲突，先说明：§3 把「语料检索引用（`references`）」列为 perceiver **刻意不做**的事，
§9 把它列为暂不做。本文改变了后者，**没有**改变前者——检索只发生在 perceiver 之后的改写阶段，
perceiver 依然不接触任何语料、也不接触 `budget`。

## 0. 两条原则

1. **只改必要的地方。** 相对 workflow，本管线只做两件事：把「严格对齐音节」的职责从 writer 搬到 mimic；
   给改写阶段一份外部诗句。perceiver 与 reviewer 一行不改（reviewer 的 7 个错误码全部保留）。
2. **检索必须是可审计的事实。** mimic 的检索部分不调用模型，但同样要落盘：查询文本、命中的 chunk 与分数、
   索引版本。没有这些，"这一版为什么更好"就无法归因。

## 1. 管线

```text
perceiver ──analysis──▶ writer ──D0──▶ mimic.检索 ──refs──▶ mimic.写作 ──D1──▶ reviewer
（整曲一次）           （每段一次）    （每段一次，不调模型）      ▲                     │
                                                               │  不合格：只回写作      │
                                                               └───────────────────────┘
                                                                    refs 不重新检索
```

`mimic` 是一个 agent、两个部分：**检索**与**写作**。两者都在 `sections/NNN/` 下各占一个目录。

与 workflow 的差异：

| | workflow | workflow-RAG |
|---|---|---|
| writer | 两条管线**共用同一个放松后的** `prompts/writer.md` | 同左 |
| 严格字数由谁承担 | reviewer 的返修意见（回 writer） | mimic 的提示词（从原 `b.md` 原样搬来） |
| 外部语料 | 无 | mimic 写作阶段拿到检索到的诗句 |
| 返修对象 | writer | mimic 写作部分（检索结果复用） |
| perceiver / reviewer | — | 不变 |

> **注意**：`writer.md` 放松之后两条管线都用它，因此 workflow 组的字数约束只剩 reviewer 的返修意见，
> 比原先"提示词里写死硬约束"要弱。换句话说，本方案相对 workflow 唯一增加的自变量是
> **mimic（严格改写 + 诗句参考）**。若以后要单独考察 writer 提示词的严格程度，那是另一个自变量，
> 需要另立一组（本轮不做）。历史 run 里仍保留当时的 `prompt_hash`，血缘不会丢。

## 2. 角色

| agent | 调用 | 输入 | 输出 | 温度 |
|---|---|---|---|---|
| **perceiver** | 整曲 1 次 | 原文 + 段落结构（不含 `budget`） | `Analysis`（字段不变） | 高（发散） |
| **writer** | 每段 1 次 | 全曲 `phrases` + `analysis` | 初版 `Draft`（`{id, zh}`，无 `chars`） | 高（初稿自由） |
| **mimic.检索** | 无（嵌入 + 向量库） | writer 本段译文 | `{query_text, k, hits[]}` | — |
| **mimic.写作** | 每段 1..N 次 | 全曲原文 + `focus` + writer 初版 + `refs` + 上一版 + 评审意见 | 第二版 `Draft`（`{id, zh, chars, self_check}`） | **1.5（默认）** |
| **reviewer** | 无（规则） | mimic.写作 的产物 | `Review`（规则不变） | — |

**一句话原则：严格性不是取消，是搬家。** 改名前 `prompts/b.md` 里「硬约束：字数」那一整节
（汉字数必须落在 budget 内、标点空格不算、不加「啊/哦」凑字、句尾避开「了/着」）**原文一字不改地搬进
mimic**。在 workflow-RAG 里它完整保留；在 workflow 里它不再出现在任何提示词中，
只剩 reviewer 的返修意见——这正是两组要对比的差别。

## 3. 提示词的搬家

提示词仍然只有「共享总提示词 + 角色提示词」两层（`dubbing/prompting.py` 的 `system_for`）。

| 文件 | 变化 |
|---|---|
| `prompts/shared.md` | 共用，不动内容；仅补一句：字数由改写的那个人负责 |
| `prompts/perceiver.md` | 原 `a.md`，内容不动，只改角色名 |
| `prompts/writer.md` | 原 `b.md`，**删掉**「硬约束：字数」整节与输出里的 `chars`/`self_check` |
| `prompts/mimic.md` | 新增：搬来的硬约束节 + 「怎么用这些诗句」 |

`ROLES` 由 `("a", "b")` 改为 `("perceiver", "writer", "mimic")`。

### 3.1 writer 的软提示（替换掉那一整节）

```markdown
## 关于字数

每个乐句后面标了大概要唱几个字（`budget`），你照着这个感觉写就行，
不用精确数字数，后面还有一个人专门负责严格对齐。
```

`budget` 字段照旧留在 `data` 里——「大概」这层意思是提示词的解释，不是数据的字段
（data-model §0：数据只存事实，不存指令）。writer 的输出只要 `{id, zh}`；
`parse_draft_output` 对 `chars`/`self_check` 本来就是可选的，因此 writer 不报也不会报错。

### 3.2 mimic 的「怎么用这些诗句」

```markdown
## 怎么用这些诗句

`references` 是从现代诗语料里检索到的诗节，供你参考。参考的是**句式、意象的构造方式、语气的松紧**，
不是词句本身。要求：

- 不得整行照抄，也不要把参考里的意象原样搬过来；
- 可以先借一个句子的骨架（比如把两个不相干的名词直接并置），再往里填自己的意象；
- 参考可以拒绝使用。`references` 为空时，就当没有参考，正常完成对齐；
- **字数永远优先于模仿**：为了对上 `budget`，先改字数，再谈贴不贴参考。
```

## 4. 每一步看到什么

### 4.1 writer（与现在的 B 同形）

```json
{
  "song": {"title": "夜に駆ける", "artist": "YOASOBI"},
  "sections": [{"id": "V1", "role": "verse"}],
  "phrases": [
    {"id": "V1-01", "section": "V1", "ja": "…", "budget": {"min": 10, "max": 10}, "zh": "……"},
    {"id": "C1-01", "section": "C1", "ja": "…", "budget": {"min": 12, "max": 12}, "prev_zh": "……"},
    {"id": "C1-05", "section": "C1", "ja": "君の声がする", "keep_source": true}
  ],
  "focus": "C1",
  "analysis": {"overall": {}, "imagery_pool": [], "sections": {}}
}
```

与现在的 `writer` 的唯一差别是**提示词怎么解释 `budget`**，入参形状不变。

### 4.2 mimic.检索

不调用模型，因此没有 system 消息；`request.json` 记参数，`output.json` 记结果，`meta.json` 记血缘。

| 项 | 取值 |
|---|---|
| 查询文本 | **本段（`focus`）可译乐句的 `zh`**，按乐句顺序拼成多行；跳过 `keep_source` 与空句 |
| 编码器 | bge-large-zh-v1.5（与建库同一个模型、同一种归一化） |
| 检索 | 单条查询、单次嵌入、Chroma top-k，`k` 默认 **4** |
| 不做 | 多路查询、分数融合、关键词路、相似度阈值、每题配额（全部砍掉） |

```json
{
  "query_text": "像要沉下去 又像要融化\n把声音写成有重量的东西\n…",
  "k": 4,
  "collection": "poetry_stanza256_v1",
  "hits": [
    {"chunk_id": "顾城_gucheng/一代人#s1", "title": "一代人", "poet": "顾城",
     "text": "黑夜给了我黑色的眼睛\n我却用它寻找光明", "score": 0.7123}
  ]
}
```

两条健壮性约定：

- 查询文本为空（整段没有可译乐句，或 writer 解析失败）→ **跳过检索**，`hits` 记空数组，
  mimic 写作阶段无参考继续。检索失败不能拖垮流程。
- `hits` 为空也要走通，mimic 的提示词里已经写明"没有参考时正常对齐"。

检索节点仍然三个文件齐全，理由和 reviewer 一样：它是产物，不是中间变量。

### 4.3 mimic.写作

**入参 = writer 的入参 + 三样东西**（初版、参考、上一版），所以编排器可以复用同一套 `phrases` 渲染。

```json
{
  "song": {"title": "夜に駆ける", "artist": "YOASOBI"},
  "sections": [{"id": "V1", "role": "verse"}, {"id": "C1", "role": "chorus"}],
  "phrases": [
    {"id": "V1-01", "section": "V1", "ja": "…", "budget": {"min": 10, "max": 10}, "zh": "……"},
    {"id": "C1-01", "section": "C1", "ja": "…", "budget": {"min": 12, "max": 12}, "prev_zh": "……"},
    {"id": "C1-05", "section": "C1", "ja": "君の声がする", "keep_source": true}
  ],
  "focus": "C1",
  "analysis": {"overall": {}, "imagery_pool": [], "sections": {}},
  "draft": {
    "section": "C1",
    "phrases": [{"id": "C1-01", "zh": "writer 的初版"}]
  },
  "references": [
    {"chunk_id": "…", "title": "一代人", "poet": "顾城", "text": "…", "score": 0.7123}
  ]
}
```

四个文本字段的含义严格区分，不合并：

| 字段 | 含义 | 能否改 |
|---|---|---|
| `zh` | 其他段已定稿的译文（mimic 的输出） | 只读参考 |
| `draft` | writer 给的**本段初版**（`D0`） | 可改，大改也可以 |
| `prev_zh` | 本段**上一轮 mimic** 的译文（首轮无此字段） | 可改，按评审意见改 |
| `references[]` | 检索到的诗节 | 只读，可用可不用 |

`draft` 与 `prev_zh` 同时给，是因为初版里常有一两个被上一轮丢掉的意象，可能是好料；
合并成一个字段就再也找不回来了。

**全曲原文与 `focus` 是必须的**，这条是相对最初设计新增的：

- `phrases` 里是**整首歌**的 `ja`，不只有本段——参考诗句是外部材料，最容易把译词带偏；
  原文在手边，mimic 才知道哪些意象是必须守住的、哪些可以换。
- `focus` 明确"你只对这一段负责"，其余段落的 `zh` 只读。
- 原曲里"连着两句完全相同"的重复美（`prompts/writer.md` 已有此要求）常常跨段落，
  只看一段会把它改坏。

### 4.4 reviewer

输入换成「写词人最终那一版」的产物：workflow 管线里是 `writer` 的，rag 管线里是 `mimic_write` 的。
其余不变：

- 7 个错误码全部保留（硬：`OVER_BUDGET` / `UNDER_BUDGET` / `EMPTY` / `NON_ZH` / `DUPLICATE`；
  软：`PUNCT` / `CHAR_MISMATCH`）；
- `prev_draft` / `prev_review` 指向**上一轮 mimic 的产物**，因此 `changes`、`regressions` 的逻辑一行不改；
- `keep_source`、`pattern`、`budget` 的处理都不变。

## 5. 编排与返修回路

```text
每一段：
  D0   = writer(...)                         # 整段只跑一次
  refs = mimic.检索(D0)                       # 只检索一次
  prev, prev_review = None, None
  for round in 1..max_rounds:
      D = mimic.写作(D0, refs, prev, prev_review)
      R = reviewer(D)
      if R.verdict == pass: 该段定稿
      prev, prev_review = D, R
```

1. **检索在返修循环之外。** 第一轮和第三轮拿到的 `refs` 完全相同，这就是"不重新检索"的语义；
   日志里也因此能把「检索质量」和「改写能力」分开看。
2. **返修信息只进消息文本，不进 `data`。** 沿用现在的做法：评审结论渲染成消息的一段说明，
   `data` 仍然只放事实。
3. **失败模式不变。** writer 初稿偏得离谱时，mimic 三轮也可能救不回来，段落终态仍是 `exhausted`，
   与现在三轮不过的终态一致。
4. **`--writer-retry` 默认关闭**（可选开关）：`exhausted` 之后再回 writer 一次，带上评审意见。
   默认关闭，是为了不把"多轮"这个变量搞浑。
5. **`--skip-done` / `--only` 要整份抄 4 个记录**：`writer`、`mimic_lookup`、`mimic_write`、`reviewer`，
   维持"每个 run 都是完整快照"的约定。
6. **`--continue`** 的语义自然变成"以上一 run 的 mimic 译文为起点"。

## 6. 目录与录制

```text
data/songs/<song_id>/
  song.yaml
  workflow/                       # 现役管线（历史 run 已迁移，见 8.2）
  workflow_rag/                   # 本方案，与 workflow 同构的兄弟目录
    latest.json
    run-<UTC 时间戳>-<8 位随机>/
      perceiver/{request,output,meta}.json          # 整曲一次
      sections/001/writer/{request,output,meta}.json
      sections/001/mimic_lookup/{request,output,meta}.json
      sections/001/mimic_write/{request,output,meta}.json
      sections/001/reviewer/{request,output,meta}.json
      sections/002/…
      lyrics.zh.txt / lyrics.md
```

- **最终文本的来源按管线不同**：workflow 读 `writer`，workflow-RAG 读 `mimic_write`；两者的
  reviewer 都读 `reviewer`。`view` / `export` / `runs` / `report` 因此只需要一个 `pipeline` 参数，
  不必为这条管线写第二套装配逻辑。
- 多轮返修同样只保留最终一轮，返修经过仍能从该轮 request 的反馈文本与 `prev_zh` 看到。
- `data/songs/*/workflow_rag/` 与 `data/poetry/` 要加进 `.gitignore`（沿现役 run 记录不入库的约定）；
  索引的 `manifest.json` 例外，它很小且是重要血缘，建议入库。

## 7. 检索资产：构建全流程

### 7.1 数据集实况

[sheepzh/poetry](https://github.com/sheepzh/poetry)：华语现代诗歌语料库，3k+ 诗人、80K+ 首诗、15M+ 字。
按诗人分目录，**一首诗一个文件**，扩展名 `.pt`，UTF-8：

```text
data/<诗人名>_<诗人拼音>/<诗标题>.pt
```

文件内部（见仓库的 [data/README.md](https://github.com/sheepzh/poetry/blob/master/data/README.md)）：

```text
title:稻草人
date:20030330
                        ← 空行分隔 header 与正文
一群鸟在离开田野好奇地
望着你，你好像无边无涯
初冬的轮子和寒霜
问你是否一起回家
                        ← 空行 = 诗节（stanza）边界
稻草人说，我可以
…
```

两个要点：诗题天然是文件名和 `title:` 字段；**正文的空行就是诗节边界**，切块规则在这个语料上直接可用。
实测一个 stanza 约 30–40 字，整首诗的规模在几行到几十行之间。

### 7.2 预处理

`data/<诗人>_<slug>/*.pt` → 解析 → 清洗 → 过滤 → 去重 → **中间产物**。

- 解析：第一个空行前是 header（`title:` / `date:`），之后是正文；诗人名取目录名最后一个 `_` 之前的部分。
- 清洗：CRLF 归一、去掉行尾空白、NBSP 与全角空格归一、去掉首尾空行。**标点全部保留**——
  现代诗的标点参与断句和语气，是我们要学的东西。
- 过滤：正文过短（< 10 字）、正文为空、缺 `title`、非中文占比过高的条目。
- 去重：正文归一化后取 sha1，砍掉完全重复（这个语料跨诗人转载不少）。
- 中间产物：`data/poetry/chunks.jsonl`，一行一个 chunk。**这是关键**：以后调切块参数、换嵌入模型、
  重建 Chroma 都不必重新解析语料。

### 7.3 切块

| 规则 | 取值 |
|---|---|
| 最小完整单元 | 诗节（stanza），即正文里被空行分开的一段 |
| 一个 chunk | 正好一个 stanza，**不设重叠** |
| token 上限 | **256**（用 bge 自己的 tokenizer 数，不是 `dubbing/counting.py` 的可唱性计数） |
| metadata | `title`（必须）、`poet`、`poem_id`、`stanza_idx`、`n_parts`、`chars`、`tokens`、`date` |
| 向量内容 | **只放 stanza 正文**，`title` 只进 metadata |
| 行内换行 | 保留。现代诗的断行本身就是句式，拍平成一行就丢了最值得学的东西 |

`title` 不进向量，是因为现代诗的标题常很抽象，塞进去会主导短 stanza 的向量；它作为 metadata 展示给
mimic（`references[].title` 已经有这一项）。

三个边界情况：

1. **整个文件没有空行**（散文诗、部分长诗）→ 整首诗就是一个 stanza，256 上限会真的触发；
2. **stanza 超过 256 token** → 按**行**贪心打包，每块不超过 256；
3. **单行超过 256 token**（极少，通常是语料脏）→ 再按 `，。！？；：、` 硬切。

拆出来的块共享同一份 metadata，另加 `stanza_id` / `part_idx` / `n_parts` / `continued: true`，
让"命中一个分块"能认出它属于同一节。实测 stanza 多在 40 token 上下，这个降级路径大部分时候不会触发，
但必须写——否则遇到一部长诗就会静默丢内容。

### 7.4 向量怎么算：两台机器分工

本机**不需要 GPU，也不需要起任何嵌入服务**。向量在服务器上算好，用文件来回传：

```text
本机                                   服务器
────                                   ────
① 切块 → chunks.jsonl
   仓库 + chunks.jsonl  ──上传──▶
                                       ② embed_poetry.py
                                          （GPU 编码；依赖见 requirements.txt）
                     ◀──下载──         vectors.npy + vectors.json
③ build_poetry_index.py --vectors … → 本地 Chroma
```

目录组织（`--out` 默认 `data/poetry`）：

| 文件 | 谁产出 | 去向 |
|---|---|---|
| `raw/` | 本机 `git clone` 语料 | 只在服务器上需要一起传 |
| `chunks.jsonl` | 本机切块（`--dry-run` 也会写） | **上传**给服务器 |
| `vectors.npy` | 服务器 `embed_poetry.py` | **下载**回本机 |
| `vectors.json` | 同上，配对说明 | 与 `.npy` 一起下载 |
| `chroma/` | 本机灌库 | 本机 |
| `manifest.json` | 本机灌库 | 本机，建议入库 |

```bash
# 上行：仓库（脚本要用）+ 切块文件
rsync -a --exclude data --exclude .git ./ user@host:~/dubbing/
rsync -a data/poetry/chunks.jsonl user@host:~/dubbing/data/poetry/

# 服务器：先建环境装依赖（一次），再编码（先 --limit 2000 验证，再全量）
ssh user@host
cd ~/dubbing
python -m venv .venv-embed && source .venv-embed/bin/activate
pip install -r requirements.txt
python scripts/embed_poetry.py --chunks data/poetry/chunks.jsonl --limit 2000 \
    --out data/poetry/vectors.head.npy --device cuda
python scripts/embed_poetry.py --chunks data/poetry/chunks.jsonl \
    --out data/poetry/vectors.npy --device cuda --batch 64

# 下行
rsync -a user@host:~/dubbing/data/poetry/vectors.npy  data/poetry/
rsync -a user@host:~/dubbing/data/poetry/vectors.json data/poetry/

# 本机灌库 + 抽查 5 条（确认服务器与本机算出来的向量一致）
python scripts/build_poetry_index.py --vectors data/poetry/vectors.npy --verify 5 --reset
```

几个要点：

- **`vectors.json` 里记着 `chunks.jsonl` 的 sha1**，灌库时会核对。换过切块参数却没重跑向量，
  会当场报错而不是悄悄用错位的向量——这是这条分工里最容易犯的错。
- **`--dtype float16`** 让文件小一半（26 万条 × 1024 维：fp32 约 1.07 GB，fp16 约 535 MB）。
  Chroma 用 cosine 距离，排序不受 dtype 影响。
- **`--verify N` 值得跑一次**：它用本机现算的向量和文件里的比余弦相似度。两边模型或池化不一致时
  检索会静默变差，这是唯一能提前发现的地方。
- 服务器上的依赖只有 `requirements.txt` 里那两行（`numpy` + `sentence-transformers`，torch 会被带进来）；
  模型默认在第一次运行时下载，国内网络可以 `export HF_ENDPOINT=https://hf-mirror.com`，
  离线机器可以先把模型目录拷过去再用 `--model /path`。
- 实测：26 万条 × 1024 维 fp16，本机灌库约 5 分钟（约 860 块/秒）。

**备选**：本机能连到 GPU 服务器时，也可以不起 TEI/vLLM 之外的任何东西——不给 `--vectors`，
直接让脚本边切边调一个 OpenAI 兼容的 `/v1/embeddings`（`DUBBING_EMBED_URL=…`）。两种模式的索引没有区别。

### 7.5 查询期也要同一个模型

建库的向量来自服务器，但**查询**（每段 1 次）是在本机发生的，必须用同一个模型：

- 本机装 `sentence-transformers`（`pip install -e ".[embed]"`），设 `DUBBING_EMBED_LOCAL=1`；
- `DUBBING_EMBED_MODEL` 指向 HF 名或模型目录，Apple 芯片可以 `DUBBING_EMBED_DEVICE=mps`；
- 每首歌十几条短查询，CPU 也够（几百毫秒一条）。

参数上一律 **CLS pooling + L2 normalize**，**不使用 instruction**——bge v1.5 的设计目标就是
"无需 instruction 也能检索"（[BGE 文档](https://bge-model.com/bge/bge_v1_v1.5.html)）。
真要加 instruction，规则是**只给查询加、chunk 不加**，并且写进 manifest。

### 7.6 Chroma（本地部署）

- **形态**：`chromadb.PersistentClient`，**本地目录**，默认 `data/poetry/chroma`。不启服务端、
  不用 HttpClient。
- **内存**：26 万 × 1024 维，索引常驻约 1–2 GB。磁盘/内存紧张时用 `--limit` 建子集索引。
  灌库走 `mmap`，本机内存占用与向量文件大小无关。
- **Collection**：`metadata={"hnsw:space": "cosine"}`；
  **`embedding_function=None` 并显式传入向量**——Chroma 默认会加载一个英文 MiniLM 帮你嵌，
  这是最经典的翻车点。
- **id**：`<诗人目录>/<诗slug>#s<节号>#p<分块号>`，人可读、可反查。
- **metadata 只放标量**（Chroma 的限制）；列表要存就序列化成字符串。
- **依赖**：`chromadb` 在 `[rag]` 可选依赖里（`pip install -e ".[rag]"`）；本机/服务器上要跑嵌入时
  再装 `[embed]`。chromadb 1.x 依赖 `onnxruntime`，目前没有 Python 3.15 的轮子（3.12 / 3.14 可以）；
  我们不用 Chroma 自带的嵌入函数，所以 `pip install --no-deps chromadb` + 手工装其余依赖也能绕开，
  但正式环境建议用有 `onnxruntime` 轮子的 Python 版本。
- **查询**：单次嵌入、top-k 默认 **4**、不设阈值、不做融合；一个 stanza 约 40 token，4 条约 160 token。
  命中分数原样落盘（`hits[].score`），供事后看检索质量。

### 7.7 `index_id` 与 manifest

索引目录下写一份 `manifest.json`：

```json
{
  "index_id": "poetry_stanza256_bge-large-zh-v1.5_v1",
  "source": "vectors",
  "corpus": {"chunks_file": "…/chunks.jsonl", "chunks": 261424, "sha1": "…"},
  "model": {"name": "BAAI/bge-large-zh-v1.5", "dim": 1024,
            "pooling": "cls", "normalize": true, "instruction": null},
  "chunking": {"unit": "stanza", "max_tokens": 256, "overlap": 0,
               "tokenizer": "transformers:BAAI/bge-large-zh-v1.5"},
  "stored": 261424,
  "built_at": "…"
}
```

``source`` 取 ``vectors``（服务器出向量）或 ``http`` / ``local``（边切边嵌）。
走 `--vectors` 时 `corpus.path` 换成 `chunks_file` + `sha1`，因为这条路径根本不读语料。

**每一条 `mimic_lookup/meta.json` 都要记 `index_id`。** 两个月后要解释两次 run 的差异，
唯一能依靠的就是它。

### 7.8 调试命令

```bash
# 有本机模型时（--embed-local）
DUBBING_EMBED_LOCAL=1 python -m dubbing poetry-search "末班车的尾灯" --k 4
# 或者仍然走一个 OpenAI 兼容端点
python -m dubbing poetry-search "末班车的尾灯" --k 4 --embed-url http://gpu:8080/v1/embeddings
```

它直接打印命中诗句与分数。调 `k`、试查询写法、抽查召回质量都靠它——否则只能靠猜。

## 8. 改名与迁移

### 8.1 代码与提示词

| 位置 | 改动 |
|---|---|
| `prompts/` | `a.md` → `perceiver.md`；`b.md` → `writer.md`；新增 `mimic.md`；`shared.md` 补一句 |
| `dubbing/prompting.py` | `ROLES = ("perceiver", "writer", "mimic")` |
| `dubbing/agents.py` | `build_a_request` → `build_perceiver_request`；`build_b_request` → `build_writer_request`；新增 `build_mimic_request`、`build_lookup_request`；`parse_b_output` → `parse_draft_output`（writer 与 mimic **共用同一个宽容解析器**） |
| `dubbing/orchestrator.py` | agent 目录名、`Step` 标签、`_meta()` 的键、新增检索节点、`workflow_rag` 的 store base |
| `dubbing/view.py` | `load_draft` 读 `mimic_write`，`load_review` 读 `reviewer` |
| `dubbing/report.py` | 录制清单、表头 |
| `dubbing/cli.py` | `--agent` choices、`--prompt`、表头、进度标签、`--temperature-{perceiver,writer,mimic}`、`--pipeline {workflow,rag}` |
| `dubbing/llm.py` | `DUBBING_TEMPERATURE_{PERCEIVER,WRITER,MIMIC}` |
| `.gitignore` | 加 `data/songs/*/workflow_rag/` 与 `data/poetry/` |
| `tests/` | 涉及 agent 名的用例 |
| `README.md`、`docs/data-model.md`、`docs/usage.md`、`baseline/PAPER_BASELINES.md`、`docs/paper-baseline-research.md`、`dubbing/store.py` 的注释 | 已同步改名 |
| `baseline/run_baselines.py` | `one-shot` 不再单独维护旧提示词：它直接用 `prompts/writer.md`，产物目录也跟着改成 `writer/`（它本来就不是 workflow-RAG 的对照） |

### 8.2 run 目录迁移脚本

`scripts/migrate_run_agents.py`，把磁盘上已有的 run 目录改成新名。**这是一次语义映射，不是改名**：

| 旧 | 新 | 理由 |
|---|---|---|
| `A/` | `perceiver/` | 同一角色改名 |
| `sections/NNN/B/` | `sections/NNN/writer/` | 同一角色改名；在 workflow 管线里它仍然是**最终被评审的那一版**，`view.load_draft` 读的就是它 |
| `sections/NNN/C/` | `sections/NNN/reviewer/` | 同一角色改名 |

所以这是一次**纯改名**：旧 run 全在 `workflow/` 下，而 `workflow/` 的最终文本位置就是 `writer/`。
（`mimic_lookup` / `mimic_write` 只存在于新管线，旧 run 里没有、也不需要有。）

一个语义上的落差要记住：旧 `B/` 是"严格的第一版"，而新 `writer/` 用的是放松后的提示词。
读旧 run 时这不成问题——读的只是"最终文本在哪儿"；血缘仍由各自的 `meta.json.prompt_hash` 说明。

其余约定：

- 迁移后 `writer/` 与 `mimic_lookup/` 在旧 run 里不存在，这是正确的——视图只读 `mimic_write` 与
  `reviewer`，照常工作；报告里这两项显示为缺失。
- `meta.json` 里的 `agent` 字段一并改写成新名（`RunStore._meta` 会写入这个字段）；
  `request.json` 里的历史文本（例如 reviewer 的 "(C 是规则检查，不调用模型)"）**保持原样**，不改写历史。
- 参数：`--root`（默认 `data/songs`）、`--song` / `--all`、`--dry-run`、`--include-baselines`（默认关闭）。
- 只处理 `run-*` 目录，`preview-*` 不动（它们只是请求快照）。
- 幂等：目标已存在就跳过并计数；任何冲突只报告、不覆盖。

## 9. 配置项

| 项 | 载体 | 默认 |
|---|---|---|
| 启用本管线 | `--pipeline rag` | `workflow` |
| perceiver 温度 | `--temperature-perceiver` / `DUBBING_TEMPERATURE_PERCEIVER` | 沿用改名前的 A 值 |
| writer 温度 | `--temperature-writer` / `DUBBING_TEMPERATURE_WRITER` | 沿用改名前的 B 值 |
| mimic 温度 | `--temperature-mimic` / `DUBBING_TEMPERATURE_MIMIC` | **1.5** |
| 检索条数 `k` | `--rag-k` / `DUBBING_RAG_K` | **4** |
| 本地索引目录 | `--rag-db` / `DUBBING_RAG_DB` | `data/poetry/chroma` |
| collection | `DUBBING_RAG_COLLECTION` | `poetry_stanza256_v1` |
| 查询期嵌入（本机） | `--embed-local` / `DUBBING_EMBED_LOCAL=1` + `DUBBING_EMBED_DEVICE` | 关（默认走下面的 HTTP 端点） |
| 查询期嵌入（HTTP） | `--embed-url` / `DUBBING_EMBED_URL` | `http://127.0.0.1:8080/v1/embeddings` |
| 嵌入模型名 | `--embed-model` / `DUBBING_EMBED_MODEL` | `BAAI/bge-large-zh-v1.5` |
| 建库：离线向量 | `--vectors` + `--chunks`（`build_poetry_index.py`） | 不给就边切边嵌 |
| 建库：向量自检 | `--verify N` | 0（关；建议首次建库时跑 5） |
| 建库：切块上限 | `--max-tokens` | 256 |
| 跨段去重 | `--rag-dedupe-peers` | 关（同一 chunk 全曲只用一次；开它会改变召回，需在报告里注明） |
| writer 失败重试 | `--writer-retry` | 关 |

服务器那边的入口就是 `scripts/embed_poetry.py`，它只依赖 `requirements.txt` 里的
`numpy` + `sentence-transformers`（torch 会被带进来），环境自己建。

## 10. 评测

**本轮不实现**，留待单独做。这里只记下它要回答的问题：检索到底有没有用、参考诗句会不会被照搬、
mimic 的对齐能力有没有被参考带坏、两条管线在同等预算下谁更好。相关指标（每段轮次与终态、
mimic 的硬违规构成、调用次数与 token、命中分数分布、与参考的最大连续重合汉字数、真人 pairwise 盲评）
以后再单独成文。
