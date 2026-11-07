# 数据模型

本文是数据与文件形式的规范，代码在 `dubbing/`（实现与本文一一对应）。

## 0. 一条原则

**数据只存事实，不存指令。**

原文、分句、音节预算、已经产出的文本，是事实；"该怎么写""别写什么"，属于提示词。要往 schema 里加字段，先问它是不是在替 B 做决定；如果是，就放进 `prompts/`。

## 1. 主干：一张乐句表

"一句歌词"在本项目里指**一个乐句**，每个乐句由人工标注一个音节数（`budget`）。因此层次只有两层：

```
song → sections → phrases
```

全流程共用同一份乐句列表，用 `phrase_id` 对齐：

| phrase_id | section | ja | budget | A 的分析 | zh | C 的检查 |
|---|---|---|---|---|---|---|
| C1-01 | C1 | 沈むように溶けてゆくように | 12 | 按 section | … | pass |
| C1-05 | C1 | 君の声がする | null | 按 section | 君の声がする | skip |

这张表是**视图**，不是存储：真相是 `song.yaml` + 一个 run 目录，`latest.json` 只决定默认看哪一次。

## 2. L0：`song.yaml`（人工标注，agent 只读）

```yaml
song:
  id: yorunikakeru
  title: 夜に駆ける
  artist: YOASOBI
  source_lang: ja

sections:
  - {id: V1, role: verse}
  - {id: C1, role: chorus}

phrases:            # 顺序即演唱顺序
  - {id: V1-01, section: V1, ja: "…", budget: {min: 10, max: 10}}
  - {id: C1-01, section: C1, ja: "沈むように溶けてゆくように", budget: {min: 12, max: 12}, notes: "结尾拖腔"}
  - {id: C1-02, section: C1, ja: "…", budget: {min: 12, max: 12}, pattern: [3, 4, 5]}
  - {id: C1-05, section: C1, ja: "君の声がする", keep_source: true}
```

字段说明：

- `budget`：人工决定的可唱汉字数，结构固定为 `{min, max}`。**MVP 阶段 `min == max`**，退化为定值；将来放开成区间时，数据结构与 C 的比较逻辑都不变。
- `budget: null`：不设约束的乐句（纯语气词、拖长音），C 跳过。
- `pattern`：这一句在旋律上被切成几段、每段几个音，例如 `[3, 4, 5]` 表示三部分共 12 个音。
  省略即"整句一段"（等价于 `[budget.max]`）。约束：每段必须是正整数、合计必须落在 `budget`
  区间内、不能和 `keep_source` 同时出现。**目前它只是给 B 看的旋律信息，C 不检查、
  也不要求译词按它断句**——所以标错了不会让流程失败，但会误导 B。
- `keep_source: true`：本句保留原文，不翻译。见第 6 节。
- `notes`：给人看的自由文本，agent 不读。

## 3. A：情感与意象

A 的输入是**原文 + 段落结构，不含 `budget`**。给了预算它就会开始数字数、开始翻译，而这不是 A 的职责。

```json
{
  "overall": {
    "theme": "…",
    "emotional_arc": "V1 压抑 → C1 爆发 → 结尾自我放逐",
    "narrative": "（一段故事化的体会，给 B 读语感）",
    "tone_words": ["窒息感", "自毁式的温柔"]
  },
  "imagery_pool": [
    {"image": "夜行的车灯", "why": "对应夜に駆ける，给出速度感与孤独感"}
  ],
  "sections": [
    {"id": "C1", "mood": "…"}
  ]
}
```

- 全部是描述，没有一处是给 B 的指令。
- `sections[].id` 是唯一的结构耦合点，必须取自 `song.yaml`。
- 整曲一次跑完，不按段跑。
- 刻意不做的：逐句提示（会把整段语感打碎）、语料检索引用（`references`）、各类"不要…"清单（属于提示词）。

## 4. B：生成译词

### 4.1 调用粒度

按 **section** 生成。B 能看到全曲（防止前后意象撞车、语气漂移），但只对 `focus` 指定的那一段负责。

### 4.2 输入

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
  "analysis": {"overall": {}, "imagery_pool": [], "sections": []}
}
```

三个文本字段的含义严格区分：

| 字段 | 含义 | B 能否改 |
|---|---|---|
| `zh` | 其他段的定稿译文 | 只读参考 |
| `prev_zh` | 本段上一版译文（首轮无此字段） | 可改 |
| `keep_source` | 本句保留原文 | 不产出译词 |

`prev_zh` 不叫 `zh`，是因为同一段既要给出旧译文、又允许它改；一个字段承担两种含义会误导。

返修轮次还会在消息里（不在 `data` 里）附上一段 C 的结论：哪几句不合格、超几个字、怎么改。
数据仍然只放事实，这段反馈是消息文本，不落进 `song.yaml`，也不落进产物。

**返修轮次与首轮的输入形状完全一致**，只是多了若干 `prev_zh`。编排器不需要两套构造逻辑。

### 4.3 输出

```json
{
  "section": "C1",
  "phrases": [
    {"id": "C1-01", "zh": "像要沉下去 又像要融化", "chars": 12,
     "self_check": {"budget": [12, 12], "ok": true}}
  ]
}
```

- `self_check` 不是约束，是让 B 自己数一遍。C 拿自报值与规则值比一次，能提前抓出"以为写了 12 字其实 14 字"。
- 产物文件只描述结果，不描述血缘（没有 `from_rev` 之类）；血缘在每个 agent 目录的 `meta.json` 里。
- `keep_source` 的乐句不出现在输出里（见第 6 节）。

## 5. C：规则检查

按 section 检查，与 B 对齐。

```json
{
  "section": "C1",
  "draft_run": "c1_b.0011",
  "verdict": "fail",
  "violations": [
    {"phrase_id": "C1-02", "code": "OVER_BUDGET", "expected": [12, 12], "actual": 15,
     "severity": "hard", "fix": "删三个虚词"}
  ],
  "changes": {
    "rewritten": ["C1-01", "C1-02"],
    "unchanged": ["C1-03"]
  },
  "regressions": ["C1-01"]
}
```

- `code` 用固定枚举，分软硬两档：
  - 硬（计入 `verdict`）：`OVER_BUDGET` / `UNDER_BUDGET` / `EMPTY` / `NON_ZH` / `DUPLICATE`
  - 软（只提示）：`PUNCT` / `CHAR_MISMATCH`（自报字数与实际不符）
- `fix` 只活在 C → B 的那一轮消息里，不落进长期数据。
- `changes.rewritten` 包含原本合格的句子，只供人看，**不算违规**。
- `regressions`：上一轮合格、本轮不合格。它不产生新规则，但会影响 `verdict`——否则返修会让质量缓慢下滑而无人察觉。

### 5.1 漂移：不约束，但可见

B 修改有问题的句子时，顺手微调相邻句子可能更好，因此**不硬性要求**已有的合格译文保持原样。代价是段级重跑可能引入新问题，所以 C 要做增量比较，输入同时拿到「当前 draft」和「上一版 draft + 上一版 report」。

## 6. `keep_source`：保留原文的乐句

极少使用，只针对无意义语气词、或原文是英文片假名之类不适合翻译的情况。**只支持乐句级**，不做 section 级语法糖，也不考虑"整段都保留"。

规则：

1. `keep_source: true` 的乐句，`budget` 必须为 `null`。同时填了两者是自相矛盾的标注，加载 `song.yaml` 时**直接报错**——这类矛盾几乎总是复制粘贴残留，让它当场暴露比悄悄忽略更省事。
2. **不由 B 产出**。B 在输入里能看到这句原文（需要它才能写出衔接），但它不是产出对象。
3. 装配视图时 `zh := ja`。
4. C 跳过该句，`verdict` 记为 `skip`。
5. 解析 B 的输出时，若出现这些 id，丢弃并记 warning。这是解析层的兜底，不是 C 的规则。

这样"不要改动"是结构性的保证——B 根本交不出这一行——不需要写进提示词去请求 B 遵守。

## 7. 视图

装配后每行统一给出"唱出来是什么"，避免下游到处判断：

```json
{"id": "C1-05", "section": "C1", "ja": "君の声がする",
 "lang": "ja", "sing_text": "君の声がする", "budget": null, "verdict": "skip"}
```

`lang` 取 `zh` / `ja`，`sing_text` 是对应的演唱文本。

## 8. 文件组织

每首歌目录下只有 `song.yaml` 和两个并列的子目录。workflow 与各 baseline **共用同一套 run 目录布局**，将来做 `workflow_v1` 也只是同层的兄弟目录。

```
data/songs/<song_id>/
  song.yaml                       # L0，人工，agent 只读
  workflow/                       # 我们自己的 A→B→C 流水线
    latest.json                   # 只回答“现在看哪一次”
    run-<UTC 时间戳>-<8 位随机>/
      A/{request,output,meta}.json                    # A 整曲一次
      sections/001/B/{request,output,meta}.json       # 这一段最终一轮
      sections/001/C/{request,output,meta}.json
      sections/002/...
      lyrics.zh.txt / lyrics.md
  baselines/                      # 对照组，结构完全一样
    one-shot/                     # workflow 提示词 + 单次调用，无 A/C
    paper-original/
    paper-annotated/

prompts/
  shared.md                       # A / B 共用的总提示词
  a.md
  b.md

corpus/                           # 预留：A 的华语歌词参考语料
evals/                            # 预留：真人评分
```

- **一次运行 = 一个 run 目录**，重跑新建目录，旧的不动。
- 每个 agent 目录固定三个文件：`request.json`（调用前就落盘）、`output.json`（解析后的产物）、`meta.json`（血缘；模型原始正文在 `raw_response` 字段里）。
- `sections/NNN` 的 `NNN` 是段落在 `song.yaml` 里的 1 起序号；段落 id 本身在 `output.json`（B 的 Draft、C 的 Review 都带 `section`）里。
- 多轮返修只保留最终一轮；返修经过仍能从该轮 request 的反馈文本与 `prev_zh` 看到。

不用"一首歌一个大 JSON"：人工标注与 agent 产物混写会带来覆盖风险和 diff 噪声，也无法并行跑多个 run 对比。

### 8.1 `latest.json`

```json
{"run": "run-20261105T124213284041Z-6a094698", "updated_at": "…", "song": "kotozute", "status": "completed"}
```

它不参与拼装视图——一个 run 目录自带全部产物，`view` / `export` / `runs` 只是默认看它指向的那一次。回滚就是改这个指针（或直接 `--run <id>`）。

### 8.2 与 LangGraph 的对应

编排器的状态只有“当前 run、当前 section、第几轮”，形状与 LangGraph 的 State 一一对应；大块文本留在磁盘上，checkpointer 不会膨胀。换成 LangGraph 时，把 `ensure_analysis` / `attempt_section` 两个函数换成节点即可。

## 9. MVP 范围

做：

- `song.yaml` + 一个 workflow run 目录（A 整曲输出 + B 段级生成 + C 的两条规则 `OVER_BUDGET` / `UNDER_BUDGET`）
- `keep_source`（含在 MVP 里，因为它会影响 B 的输入契约，事后加要改契约结构）
- `pattern`（同样影响 B 的输入契约，先只透传、不约束）

暂不做（加进去不改变结构）：

- `budget` 放开成真正的区间（MVP 里 `min == max`）
- 跨行借还字数（`borrow`）
- `pattern` 的强制检查（现在只透传给人看）
- A 的语料检索（`references`）
- 原文 ↔ 译文的行内词对齐
