# 怎么跑

数据模型见 [data-model.md](data-model.md)。这篇只讲操作。

## 环境

需要 Python ≥ 3.11。基础运行依赖只有 `pyyaml` 和 `requests`；测试用 `pytest`。
rag 管线另外两组可选依赖，按需装：

- `.[rag]` —— `chromadb`，本地向量库（跑 rag 管线、`poetry-search` 都要）；
- `.[embed]` —— `sentence-transformers`，在本机算**查询**向量（只灌库不需要它；只走 HTTP 端点也不需要）。

```bash
conda activate dubbing          # 或你自己的虚拟环境
python -m pytest                # 280 个测试，1 秒左右
pip install -e ".[dev,rag,embed]"   # 只跑 workflow 的话 ".[dev]" 就够
```

## 三条命令

```bash
python -m dubbing songs                     # 歌库里有什么
python -m dubbing run     --song kotozute   # 跑一首（带进度）
python -m dubbing run     --all             # 跑歌库里所有歌
python -m dubbing runs    --song kotozute   # 所有 run，标出 latest.json 指的是哪一次
python -m dubbing prompt  --song kotozute   # 看 writer 收到的提示词与数据
python -m dubbing show    --song kotozute   # 看 perceiver / writer / reviewer 各自的产物
python -m dubbing view    --song kotozute   # 调试视图（每句状态）
python -m dubbing export  --song kotozute   # 导出成品歌词
python -m dubbing poetry-search "一句中文" --k 4 --embed-local   # 直接查本地现代诗索引
```

### 看某个 agent 收到的提示词

```bash
python -m dubbing prompt --song kotozute --agent writer --section B1
python -m dubbing prompt --song kotozute --agent writer --section B1 --run <run_id>  # 某次真实调用
```

`prompt` 打印两段：`system`（`prompts/shared.md` + `prompts/writer.md` 拼起来的，约 2100 字）和
`user`（这次要喂的数据，约 6-8k 字）。

- 不加 `--run` 是**现拼**：按当前 `song.yaml` 和上一 run 算出的“如果现在调用会发什么”，
  返修反馈不会出现（没有“上一轮”）。
- 加 `--run` 是**原样回放**：读那个 run 的 `sections/NNN/writer/request.json`（用 `--agent` /
  `--section` 选具体哪一条），包括返修轮里 reviewer 的结论。
- 只想看提示词文本本身，不看数据：`prompts/shared.md` 和 `prompts/writer.md`，顺序拼起来就是。
- `--agent` 取 `perceiver` / `writer` / `mimic`；`mimic` 读的是 rag 管线里**写作那次**请求
  （检索那次见 `show --agent mimic_lookup`）。
- `--pipeline` 决定读哪条管线，默认 `workflow`，也可以用环境变量 `DUBBING_PIPELINE` 设默认值。

### 看每个 agent 的产物

`show` 默认把 perceiver、各段的 writer（rag 管线还有各段的 mimic_lookup / mimic_write）、
各段的 reviewer 全部打印出来；也可以只挑一个：

```bash
python -m dubbing show --song kotozute --agent perceiver                   # 只有 perceiver 的体会
python -m dubbing show --song kotozute --agent writer --section C1         # 只有 C1 的初版译词
python -m dubbing show --song kotozute --pipeline rag --agent mimic_lookup --section C1  # C1 拿什么去检索、命中了什么
python -m dubbing show --song kotozute --pipeline rag --agent mimic_write --section C1   # C1 的最终译词
python -m dubbing show --song kotozute --agent reviewer --section C1       # 只有 C1 的检查结论
python -m dubbing show --song kotozute --run <run_id>                      # 直接看某一次 run
```

`--agent` 的取值是 `perceiver` / `writer` / `mimic_lookup` / `mimic_write` / `reviewer`。
几种渲染各有侧重：perceiver 是散文（主题 / 情绪走向 / 意象池 / 叙事）；写词的人（workflow 的
writer、rag 的 mimic_write）是表格（原文 / 译词 / 字数 / 预算），mimic 的自查说明附在表下；
reviewer 是结论（verdict / 违规 / 改了哪些 / 有没有回归）；mimic_lookup 单独渲染成
「查询 + 命中诗句 + 索引血缘」。

`runs` 列出这首歌的每一次运行，`*` 标的是 `latest.json` 指向的那一次：

```
   run_id                         模型            perceiver  段落      歌词  tokens  状态
*  run-20261103T173709Z-eb63194f  deepseek-flash  有         4/4 pass  有    2286    completed
```

### 批量跑

```bash
python -m dubbing run --all --client api               # 每首各新建一个 run
python -m dubbing run --song a --song b --client api   # 指定几首
python -m dubbing run --all --dry-run                  # 只渲染请求，不调用
```

- 每首歌**独立**新建自己的 run 目录，互不影响；
- 一首失败不会中断整批，最后打印 `N 首成功 / M 首失败` 和失败清单；
- 例外是 exit 2（需要 API / 需要人工），那种情况直接停下，因为后面的歌会以同样原因失败；
- 顺序是 `songs` 的排序，逐首打印 `=== [3/10] kotozute ===` 这样的分隔。

先估成本：`--all --dry-run` 会告诉你每首几个请求（当前 10 首共 49 个请求）。

### 跑的时候会打印进度

`run` 把每一步打到 **stderr**（stdout 留给最终结果，方便管道）：

```
[perceiver] 体会整首歌 …
[perceiver] 4 段
[writer] V1 第 1 轮 …
[reviewer] V1 pass（0 硬 / 0 软）
[writer] V2 第 1 轮 …
…
[writer] C1 第 2 轮 …
[reviewer] C1 pass（0 硬 / 0 软）
```

rag 管线在 writer 的初版之后多两步（检索一次，之后每轮改写都用同一批参考）：

```
[writer] V1 出初版 …
[mimic] V1 检索到 4 条诗句
[mimic] V1 改写 第 1 轮 …
[reviewer] V1 pass（0 硬 / 0 软）
```

返修时轮次会往上加，一眼能看出卡在哪一段。

成品歌词有两个文件，`run` 跑完会写进这一次的 run 目录：

- `data/songs/<id>/workflow/<run_id>/lyrics.zh.txt` —— 纯中文，按段落分开，直接唱或复制
  （rag 管线的目录是 `workflow_rag/`，下同）；
- `data/songs/<id>/workflow/<run_id>/lyrics.md` —— 原文 / 译词 / 字数 / 状态对照表，校对语义用。

只跑某一段（省 token，改一句就重跑一段）：

```bash
python -m dubbing run --song kotozute --client api --only C1
```

`run` 的几种语义，别搞混：

| 命令 | 旧 run | perceiver | writer / reviewer |
|---|---|---|---|
| `run`（默认） | 保留在盘上 | 复用 | 全部重跑 |
| `run --redo-analysis` | 保留 | 重跑 | 全部重跑 |
| `run --continue` | 保留 | 复用 | 重跑，旧译文作为 `prev_zh` |
| `run --skip-done` | 保留 | 复用 | 已定稿的段整份抄过来 |
| `run --only C1` | 保留 | 复用 | 只重跑 C1，其余段抄过来 |

（rag 管线里 writer 后面还有 mimic：`writer`、`mimic`、`reviewer` 一起按同一规则重跑，
`--continue` 的起点是上一 run 的 mimic 译文。）

默认**新建一个 run、writer / reviewer 从头写**，但 **perceiver 复用上一 run 的产物**
（perceiver 是整曲一次，白跑一遍纯浪费）。想让 perceiver 也重新体会，加 `--redo-analysis`。

旧 run 永远不删，所以新的一跑中途失败（网络、额度）也不会破坏任何已有结果；
`latest.json` 只在整首跑完后才指向新 run。

- **pending 客户端**（没加 `--client api`）只把入参写进新 run 就停，旧 run 不受影响。
- **`--only` / `--continue` / `--skip-done`** 会把没重跑的段落整份抄进新 run，
  所以每个 run 都是完整快照。

`run` 默认**不会真的调用模型**：它把渲染好的入参写到新 run 的 `perceiver/request.json`，
然后停下并告诉你去哪里看。想先看请求长什么样（只写请求、不调用）：

```bash
python -m dubbing run --song kotozute --dry-run
```

它会写出若干对文件到 `data/songs/kotozute/workflow/preview-<时间戳>/`（rag 管线在
`workflow_rag/` 下）：

```
perceiver.system.md      perceiver 的提示词（共享总提示词 + 角色提示词）
perceiver.data.json      perceiver 的入参数据
writer-V1.system.md      writer 在某一段的提示词
writer-V1.data.json      writer 在该段的入参数据（全曲乐句 + 预算 + 焦点段）
mimic-V1.system.md       rag 管线才有：mimic 写作的提示词
mimic-V1.data.json       rag 管线才有：初版与参考都是空的（要跑完 writer 与检索才填得上）
```

## 接 API

已接好，走 DeepSeek 的 Anthropic 兼容端点：

```
POST https://api.deepseek.com/anthropic/v1/messages
x-api-key: $DEEPSEEK_API_KEY
```

key 放两处之一：

```bash
# 推荐：写进仓库根的 .env（已 gitignore），命令行启动时自动载入
cp .env.example .env && $EDITOR .env

# 或者：当前终端里 export（只对这条命令生效）
export DEEPSEEK_API_KEY=sk-...
```

然后：

```bash
python -m dubbing run --song kotozute --client api
```

`.env` 的规则：**真 key 只写在 `.env`**；`.env.example` 是模板，要入库，里面只能有占位符。

### 换接口地址

`DUBBING_BASE_URL` 默认指向 DeepSeek 的 Anthropic 兼容端点，改成自建网关或代理都行：

```bash
# 写在 .env 里
DUBBING_BASE_URL=https://my-gateway.example.com/anthropic

# 或临时覆盖
python -m dubbing run --song kotozute --base-url http://127.0.0.1:8080/anthropic
```

客户端会在根地址后拼 `/v1/messages`，所以别自己带这一段（`.../anthropic` ✓，`.../anthropic/v1/messages` ✗）。
模型名（`DUBBING_MODEL` / `--model`）要跟你的网关认得的一致；`deepseek-flash` 是 DeepSeek 的。

| 环境变量 | 作用 | 默认 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 密钥（也认 `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`） | 无 |
| `DUBBING_CLIENT` | `pending` / `api` | `pending` |
| `DUBBING_MODEL` | 模型名 | `deepseek-flash` |
| `DUBBING_BASE_URL` | 接口根地址 | `https://api.deepseek.com/anthropic` |
| `DUBBING_TEMPERATURE` | 通用温度 | `0.7` |
| `DUBBING_TEMPERATURE_PERCEIVER` | 只给 perceiver | 跟随通用值 |
| `DUBBING_TEMPERATURE_WRITER` | 只给 writer | 跟随通用值 |
| `DUBBING_TEMPERATURE_MIMIC` | 只给 mimic（rag 的改写） | `1.5` |
| `DUBBING_PIPELINE` | 默认管线 | `workflow` |

命令行可以覆盖：`--client` / `--model` / `--temperature` / `--temperature-perceiver` /
`--temperature-writer` / `--temperature-mimic` / `--max-tokens`。

### 温度：perceiver 与写词的人该分开调

perceiver 的任务是发散（把意象摊开），mimic 的任务是收敛（把字塞进 `budget`），writer 在
两者中间：它写初版，字数只要大概对上。同一个温度对谁都不合适：

```bash
# perceiver 放开，writer 收紧
python -m dubbing run --song kotozute --client api --temperature-perceiver 2.0 --temperature-writer 0.4
```

每次调用的实际温度记在该次调用的 `meta.json` 的 `temperature` 字段里（例如
`workflow/<run>/sections/003/writer/meta.json`），按 agent 覆盖的温度只影响对应 agent，不会串。
`mimic` 另有默认值 **1.5**——它要在字数硬约束下改词，比通用值松。

几个行为：

- **重试**：429 与 5xx 重试 3 次（指数退避，认 `Retry-After`）；4xx 直接报错不重试，因为那是请求本身有问题。
- **截断**：`stop_reason == max_tokens` 会明确报错，而不是拿半截 JSON 去解析。真遇到就调大 `--max-tokens` 或把 section 拆小。
- **用量**：每次调用的 `usage` 记进该次调用的 `meta.json`。
- **thinking 块**：响应里若有 `thinking` 块会被跳过，只取 `text` 块——否则推理过程会污染 JSON。
- **返修**：reviewer 不合格时，下一轮会把它的结论（哪句、超几个字、怎么改）渲染进写词人的消息，
  写词的人才知道要改什么——workflow 里回 `writer`，rag 里只回 `mimic_write`（检索结果复用，
  不重新检索）。这段反馈只存在于消息里，不写入任何数据文件。

## 跑 rag 管线

第二条管线在 perceiver → writer 之后多一个 **mimic**：用 writer 的初版译文去本地现代诗索引里
检索一批诗节，再在字数硬约束下把初版改写成最终唱词；reviewer 检查的是 mimic 的写作产物。
**严格字数归 mimic**——workflow 里则只剩 reviewer 的返修意见，这个差别正是两条管线要对比的，
方案细节见 [rag-workflow.md](rag-workflow.md)。

先建本地索引。索引分两台机器建：**本机切块 → 服务器用 GPU 出向量 → 本机灌进 Chroma**。
本机不需要 GPU，也不需要起任何嵌入服务。目录组织与参数见 [rag-workflow.md](rag-workflow.md) 第 7 节。

```bash
# ① 本机：解析 + 切块 + 统计（不联网、不建库），顺带写出 data/poetry/chunks.jsonl
python scripts/build_poetry_index.py --poetry-root data/poetry/raw --dry-run

# ② 上传：仓库（脚本要用）+ 切块文件
rsync -a --exclude data --exclude .git ./ user@host:~/dubbing/
rsync -a data/poetry/chunks.jsonl user@host:~/dubbing/data/poetry/

# ③ 服务器：自建环境装依赖（根目录 requirements.txt），再用 GPU 编码
ssh user@host
cd ~/dubbing
python -m venv .venv-embed && source .venv-embed/bin/activate
pip install -r requirements.txt
python scripts/embed_poetry.py --chunks data/poetry/chunks.jsonl --limit 2000 \
    --out data/poetry/vectors.head.npy --device cuda          # 先用 2000 条跑通
python scripts/embed_poetry.py --chunks data/poetry/chunks.jsonl \
    --out data/poetry/vectors.npy --device cuda --batch 64    # 全量

# ④ 下载向量；先抽查 5 条（不写索引），再灌库
rsync -a user@host:~/dubbing/data/poetry/vectors.npy  data/poetry/
rsync -a user@host:~/dubbing/data/poetry/vectors.json data/poetry/
python scripts/build_poetry_index.py --vectors data/poetry/vectors.npy --verify 5 --skip-ingest
python scripts/build_poetry_index.py --vectors data/poetry/vectors.npy --reset
```

几条容易忘的：

- `--limit N` 会**同时**把用到的 chunks 另存成 `chunks.limit{N}.jsonl`；灌库时要 `--chunks` 指它，
  否则条数对不上（`vectors.json` 里的 sha1 也是对着这份截断文件算的）。
- `--verify N` 用本机模型现算几条、和文件里的比余弦相似度，应当接近 1.0。它在灌库**之前**跑，
  不过就停下、不动已有索引；`--skip-ingest` 表示只验不灌。
- `--reset` 只在**换模型 / 改 `--max-tokens` / 换语料**时需要（旧 chunk_id 会残留成"不存在的诗句"）。
  同一套重跑、或从子集扩到全量（id 是超集）都不需要。
- 本机下载模型时如果被 SOCKS 代理挡住（报 `socksio` 缺失或 `Could not resolve the revision`），
  要么 `pip install socksio`，要么 `env -u all_proxy` 直连。

建好之后：

```bash
python -m dubbing run --song kotozute --pipeline rag --client api --embed-local
python -m dubbing poetry-search "一句中文" --k 4 --embed-local   # 不跑流程，直接查索引、看命中分数
```

产物在 `data/songs/<id>/workflow_rag/`，每段四个目录：

```
sections/NNN/writer/        初版（writer）
sections/NNN/mimic_lookup/  检索参数与命中诗句（不调模型）
sections/NNN/mimic_write/   最终译词（mimic 的写作部分）
sections/NNN/reviewer/      检查结论
```

检索**只做一次**，查询就是 writer 的初版译文；返修只回 `mimic_write`，不重新检索。
`runs` / `show` / `view` / `export` 都认 `--pipeline`（默认 `workflow`），也可以用环境变量
`DUBBING_PIPELINE` 设默认值。

常用选项：

| 选项 | 作用 | 默认 |
|---|---|---|
| `--rag-k` | 检索条数 | `4` |
| `--rag-db` | 本地 Chroma 目录 | `data/poetry/chroma` |
| `--rag-collection` | collection 名 | `poetry_stanza256_v1` |
| `--embed-url` / `--embed-model` | 嵌入端点与模型 | `DUBBING_EMBED_URL` / `BAAI/bge-large-zh-v1.5` |
| `--embed-local` | 本机推理（需要 sentence-transformers） | 关 |
| `--rag-dedupe-peers` | 跨段去重：已经用过的 chunk 不再召回（开了会改变召回） | 关 |
| `--writer-retry` | mimic 各轮都没过时，再让 writer 带评审意见重出初版 | 关 |

**查询期必须能算向量。** 灌库不用模型，但每次检索都要把"这一段译文"编码成向量：

- 本机模型：`--embed-local`（或 `DUBBING_EMBED_LOCAL=1`），模型默认 `BAAI/bge-large-zh-v1.5`，
  第一次会下到 `~/.cache/huggingface`（约 1.3 GB）；Apple 芯片可加 `DUBBING_EMBED_DEVICE=mps`；
- 或者仍然连一个 OpenAI 兼容端点：`--embed-url http://gpu:8080/v1/embeddings`。

两边用的必须是**同一个模型、同一种池化**，否则检索会静默变差——`--verify` 就是查这个的。

`--dry-run` 渲染不出 mimic 的真实入参（初版与参考要等 writer 与检索跑完），只能核对提示词与
字段形状。

## 旧 run 的 agent 名

改名之前建的 run 目录还在用 `A/` `B/` `C/`，用脚本一次性改过来：

```bash
python scripts/migrate_run_agents.py --dry-run --all             # 先看会改什么
python scripts/migrate_run_agents.py --all                       # workflow/ 下的所有 run
python scripts/migrate_run_agents.py --all --include-baselines   # 连 baselines/one-shot 一起
```

`--song` 可重复（与 `--all` 二选一）。脚本只动 `run-*` 目录，`preview-*` 是请求快照、不改；
目标目录已存在就跳过，不覆盖。旧 run 里没有 `mimic_lookup` / `mimic_write`，这是对的——
workflow 管线读的就是 `writer/`。

## 加一首歌

新建 `data/songs/<song_id>/song.yaml`，把日文按乐句一行一句拆开，每句标上
该唱几个字（`budget`）。格式见 [data-model.md](data-model.md) 第 2 节。

`budget` 的由来：数原句在旋律上占几个音符，再人工判断哪些音符可以合并
（连读、拖腔），得出这句"允许写几个汉字"。这一步 AI 做不了，必须人工。

如果这一句在旋律上明显分成几段（中间有换气或长停顿），可以再标一个 `pattern`：

```yaml
- {id: C1-02, section: C1, ja: "…", budget: {min: 12, max: 12}, pattern: [3, 4, 5]}
```

`[3, 4, 5]` = 三部分组成、分别唱 3/4/5 个字，合计 12。不写就是"整句一段"。
目前它只是喂给写词的人（workflow 的 writer、rag 的 writer 与 mimic）的参考信息，
reviewer 不检查，也不要求译词照着切。

### 从标注文本批量导入

如果已经按「歌词 总字数 [分句...]」的形式断好句，可以直接批量生成
`song.yaml`，不用手写 YAML。完整示例在 `data/lyrics.txt`：

```
## ソナーレ
眠りから目覚めたら 10
ひらかれたページの上にひとり 14

ばらばらのリズムで 8
口ずさめたらきいて 9
```

- 以 `#` / `##` 开头的是歌名；歌名后可写 `[song-id]` 显式指定目录名；
- 空行切分 section，section 内的每一行是一句；
- 行尾第一个数字是 `budget`（MVP 里 `min == max`），后面的数字是 `pattern`；
- `*这句不用翻译*` 表示该句 `keep_source: true`，此行可以没有数字。

```bash
python scripts/import_lyrics.py data/lyrics.txt --dry-run   # 先看解析结果
python scripts/import_lyrics.py data/lyrics.txt             # 写入 data/songs/
python scripts/import_lyrics.py data/lyrics.txt --force     # 覆盖已有 song.yaml
```

section 的 `role` 是按位置和重复关系粗略推测的（副歌、pre-chorus、outro），
导入后建议人工复核。分句合计与总字数不一致时会打印 `!` 警告并尽量修复；
不想让它猜，加 `--strict` 直接报错。

## 跑 baseline

对照组用统一入口，用 `--baseline` 选：

```bash
python baseline/run_baselines.py --song kotozute --baseline all --dry-run   # 先预览
python baseline/run_baselines.py --all --baseline paper --client api
python baseline/run_baselines.py --song kotozute --baseline one-shot --client api
```

- `one-shot`：workflow 的提示词 + 单次调用，没有 perceiver 分析、没有 reviewer 检查与返修（消融对照）；
- `paper-original` / `paper-annotated`：论文 Exp.4 的两组；后者传人工预算与分组。

输出组织与 workflow 一样，落在 `baselines/<名字>/<run_id>/`。
详见 [../baseline/PAPER_BASELINES.md](../baseline/PAPER_BASELINES.md)。

## 目录会变成什么样

每首歌目录下只有 `song.yaml` 和三个并列的子目录：

```
data/songs/kotozute/
  song.yaml
  workflow/                                  # 我们自己的 perceiver→writer→reviewer 流水线
    latest.json                              # 只回答“现在看哪一次”
    run-20261103T173709Z-eb63194f/
      perceiver/{request,output,meta}.json
      sections/001/writer/{request,output,meta}.json
      sections/001/reviewer/{request,output,meta}.json
      lyrics.zh.txt / lyrics.md
  workflow_rag/                              # perceiver→writer→mimic（检索 + 改写）→reviewer
    latest.json
    run-20261103T173709Z-eb63194f/
      perceiver/{request,output,meta}.json
      sections/001/writer/{request,output,meta}.json         # 初版
      sections/001/mimic_lookup/{request,output,meta}.json   # 检索参数与命中诗句
      sections/001/mimic_write/{request,output,meta}.json    # 最终译词
      sections/001/reviewer/{request,output,meta}.json
      lyrics.zh.txt / lyrics.md
  baselines/                                 # 对照组，结构完全一样
    one-shot/  paper-original/  paper-annotated/
```

每个 agent 目录**固定三个文件**：`request.json`（调用前就落盘）、`output.json`（产物）、
`meta.json`（模型 / 用量 / prompt_hash / 轮次；模型原始正文在 `raw_response` 里）。
`mimic_lookup` 不调模型，它的 `meta.json` 记的是检索参数与索引血缘。

一次运行一个 run 目录、永不覆盖，所以任何一版译词都能追到“哪次调用、哪版提示词、什么入参”；
两条管线各有自己的 `latest.json`，只决定 `view` / `export` / `runs` 默认看哪一次，
想换一版就改它（或直接 `--run <id>`）。

索引资产不在歌目录里，单独放在 `data/poetry/`。除了 `manifest.json`，其余都不入库：

```
data/poetry/
  raw/                    语料 clone（sheepzh/poetry）
  chunks.jsonl            切块中间产物（要上传给服务器）
  chunks.limit2000.jsonl  --limit 跑出来的子集，与 head 向量配套
  vectors.npy             服务器产出的向量（fp16，26 万条约 535 MB）
  vectors.json            配对说明：模型、维度、条数、chunks 的 sha1
  chroma/                 本地向量库（26 万条约 1.6 GB）
  manifest.json           索引血缘：index_id、模型、切块参数、建库时间
```

`manifest.json` 的 `index_id` 会被写进每一次 `mimic_lookup/meta.json`，所以任何一版译词都能追到
"它当时查的是哪份索引"。

## 几个容易踩的点

- **严格字数归谁**：writer 的提示词只说“大概几个字”，不要求精确数字数。workflow 里字数约束
  只剩 reviewer 的返修意见；rag 里由 `prompts/mimic.md` 的硬约束那一节承担。
- **perceiver 的入参里没有 `budget`**。这是故意的，perceiver 一看到字数就会开始数字数、开始翻译。
- **`zh` 和 `prev_zh` 不是一回事**。前者是别的段落已定稿的，后者是写词的人（workflow 的 writer、
  rag 的 mimic）自己上一版、可以改。顺手改好相邻句子是允许的，reviewer 会在 `changes.rewritten`
  里让你看到。
- **`PUNCT` 和 `CHAR_MISMATCH` 是软违规**，只提示不影响 `verdict`；其余都是硬违规。
  `CHAR_MISMATCH` 只在写词的人自报了 `chars` 时才有意义——现在只有 rag 的 mimic 会报。
- **`keep_source` 的句子不会出现在写词人的输出里**。装配时 `sing_text` 直接取原文，
  reviewer 跳过它。
- **灌库不需要模型，查询需要。** `build_poetry_index.py --vectors` 只读向量文件；
  但 `poetry-search` 和 rag 管线每次检索都要编码查询，所以本机得装 `.[embed]` 并加 `--embed-local`。
- **索引与向量必须同源。** `vectors.json` 记着 `chunks.jsonl` 的 sha1，条数 / 维度 / sha1 任何一项不对，
  灌库前就会报错——这正是为了防止"换了切块参数，向量还是旧的"。
- **`--reset` 不是"清空这次导入的"，是"清空整个 collection"**。换模型或改切块参数时该加；
  同一套重跑、或从子集扩到全量（id 是超集）时不该加。
