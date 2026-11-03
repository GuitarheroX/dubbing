# 怎么跑

数据模型见 [data-model.md](data-model.md)。这篇只讲操作。

## 环境

只需要 Python ≥ 3.11 和 `pyyaml`，测试用 `pytest`。

```bash
conda activate base
python -m pytest            # 44 个测试，跑完不到 1 秒
```

## 三条命令

```bash
python -m dubbing songs                     # 歌库里有什么
python -m dubbing run     --song kotozute   # 跑一首（带进度）
python -m dubbing run     --all             # 跑歌库里所有歌
python -m dubbing runs    --song kotozute   # 所有 run，标出当前用的是哪版
python -m dubbing show    --song kotozute   # 看 A / B / C 各自的产物
python -m dubbing view    --song kotozute   # 调试视图（每句状态）
python -m dubbing export  --song kotozute   # 导出成品歌词
```

### 看每个 agent 的产物

`show` 默认把 A、各段的 B、各段的 C 全部打印出来；也可以只挑一个：

```bash
python -m dubbing show --song kotozute --agent a              # 只有 A 的体会
python -m dubbing show --song kotozute --agent b --section C1 # 只有 C1 的译词
python -m dubbing show --song kotozute --agent c --section C1 # 只有 C1 的检查结论
python -m dubbing show --song kotozute --run <run_id>         # 直接看某一次 run
```

三种渲染各有侧重：A 是散文（主题 / 情绪走向 / 意象池 / 叙事），B 是表格（原文 / 译词 /
字数 / 预算 / 自查说明），C 是结论（verdict / 违规 / 改了哪些 / 有没有回归）。

`runs` 列出所有历史 run，`*` 标的是当前视图正在用的那一版：

```
   run_id                    agent  模型            轮  tokens  说明
*  20261027T143532_a_f07bdb  A      deepseek-flash      1988    4 段
*  20261027T150039_b_961b90  B      deepseek-flash  1   3605    11 句
*  20261027T150056_c_1268ec  C      -                           pass
```

### 批量跑

```bash
python -m dubbing run --all --client api          # 全部重来
python -m dubbing run --all --client api --keep-history
python -m dubbing run --song a --song b --client api   # 指定几首
python -m dubbing run --all --dry-run             # 只看规模，不调用
```

- 每首歌**独立**套用 `run` 的语义（默认是清空重来，逐首清空）；
- 一首失败不会中断整批，最后打印 `N 首成功 / M 首失败` 和失败清单；
- 例外是 exit 2（需要 API / 需要人工），那种情况直接停下，因为后面的歌会以同样原因失败；
- 顺序是 `songs` 的排序，逐首打印 `=== [3/10] kotozute ===` 这样的分隔。

先估成本：`--all --dry-run` 会告诉你每首几个请求（当前 10 首共 49 个请求）。

### 跑的时候会打印进度

`run` 把每一步打到 **stderr**（stdout 留给最终结果，方便管道）：

```
[A] 体会整首歌 …
[A] 4 段
[B] V1 第 1 轮 …
[C] V1 pass（0 硬 / 0 软）
[B] V2 第 1 轮 …
…
[B] C1 第 2 轮 …
[C] C1 pass（0 硬 / 0 软）
```

返修时轮次会往上加，一眼能看出卡在哪一段。

成品歌词有两个文件，`run` 跑完会自动写：

- `data/songs/<id>/lyrics.zh.txt` —— 纯中文，按段落分开，直接唱或复制；
- `data/songs/<id>/lyrics.md` —— 原文 / 译词 / 字数 / 状态对照表，校对语义用。

只跑某一段（省 token，改一句就重跑一段）：

```bash
python -m dubbing run --song kotozute --client api --only C1
```

`run` 的几种语义，别搞混：

| 命令 | 旧结果 | A | B / C |
|---|---|---|---|
| `run`（默认） | **删掉**（`state.yaml` + `runs/`） | 重跑 | 全部重跑 |
| `run --keep-history` | 移进 `archive/<时间戳>/` | 重跑 | 全部重跑 |
| `run --continue` | 保留 | 复用 | 重跑，旧译文作为 `prev_zh` |
| `run --skip-done` | 保留 | 复用 | 只跑没定稿的段 |
| `run --only C1` | 保留 | 复用 | 只跑 C1 |

默认是**彻底重来**：什么都不复用，A 也会重新体会一遍。适合"这一版整体不满意"。

两个保护：

- **pending 客户端不清空**。没加 `--client api` 时只渲染请求就停，这时候清空纯属白删，
  所以会直接拒绝并提示加 `--client api`。
- **`--only` / `--continue` / `--skip-done` 都不清空**。这几个本来就是"局部动手"的意思。

清空的代价是真实的：新的一跑如果中途失败（网络、额度），旧结果已经不在了。
想留退路就加 `--keep-history`，旧 run 会整批移进 `archive/<时间戳>/`，`runs` 和 `show`
里看不到它们，但磁盘上还在。

`run` 默认**不会真的调用模型**：它把渲染好的入参写到 `runs/<id>/request.json`，
然后停下并告诉你去哪里看。想先看请求长什么样（完全不产生 run 目录）：

```bash
python -m dubbing run --song kotozute --dry-run
```

它会写出四对文件到 `data/songs/kotozute/preview/`：

```
a.system.md      a 的提示词（共享总提示词 + 角色提示词）
a.data.json      a 的入参数据
b-V1.system.md   b 在某一段的提示词
b-V1.data.json   b 在该段的入参数据（全曲乐句 + 预算 + 焦点段）
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
| `DUBBING_TEMPERATURE_A` | 只给 A | 跟随通用值 |
| `DUBBING_TEMPERATURE_B` | 只给 B | 跟随通用值 |

命令行可以覆盖：`--client` / `--model` / `--temperature` / `--max-tokens`。

### 温度：A 和 B 该分开调

A 的任务是发散（把意象摊开），B 的任务是收敛（把字塞进预算）。同一个温度对两者都是错的：

```bash
# A 放开，B 收紧
python -m dubbing run --song kotozute --client api --temperature-a 1.2 --temperature-b 0.4
```

经验区间：A 用 1.0–1.3，容易出偏一点但有意思的意象；B 用 0.3–0.6，字数更稳、'自由地' 这种偷懒解更少。
默认 0.7 是两者的折中值，适合先跑通。

每次调用的实际温度记在 `runs/<id>/meta.json` 的 `temperature` 字段里，`temperature_by_agent`
的覆盖只影响对应 agent，不会串。

几个行为：

- **重试**：429 与 5xx 重试 3 次（指数退避，认 `Retry-After`）；4xx 直接报错不重试，因为那是请求本身有问题。
- **截断**：`stop_reason == max_tokens` 会明确报错，而不是拿半截 JSON 去解析。真遇到就调大 `--max-tokens` 或把 section 拆小。
- **用量**：每次调用的 `usage` 记进 `runs/<id>/meta.json`。
- **thinking 块**：响应里若有 `thinking` 块会被跳过，只取 `text` 块——否则推理过程会污染 JSON。
- **返修**：C 不合格时，下一轮会把 C 的结论（哪句、超几个字、怎么改）渲染进 B 的消息，
  B 才知道要改什么。这段反馈只存在于消息里，不写入任何数据文件。

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
目前它只是喂给 B 的参考信息，C 不检查，也不要求译词照着切。

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

## 目录会变成什么样

```
data/songs/kotozute/
  song.yaml                      人标的，agent 只读
  state.yaml                     指针：当前视图由哪几个 run 拼成（gitignore）
  runs/20261027T075359_a_088264/
    request.json                 渲染后的入参
    output.json                  产物
    meta.json                    model / prompt_hash / input_hash
  preview/                       --dry-run 的产物（gitignore）
```

`state.yaml` 只存指针，所以回滚就是改指针；`runs/` 永不覆盖，所以任何一版译词
都能追到"哪次调用、哪版提示词、什么入参"。

## 几个容易踩的点

- **A 的入参里没有 `budget`**。这是故意的，A 一看到字数就会开始数字数、开始翻译。
- **`zh` 和 `prev_zh` 不是一回事**。前者是别的段落已定稿的，后者是 B 自己上一版、
  可以改。B 顺手改好相邻句子是允许的，C 会在 `changes.rewritten` 里让你看到。
- **`PUNCT` 和 `CHAR_MISMATCH` 是软违规**，只提示不影响 `verdict`；其余都是硬违规。
- **`keep_source` 的句子不会出现在 B 的输出里**。装配时 `sing_text` 直接取原文，
  C 跳过它。
