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

这张表是**视图**，不是存储：真相是 `song.yaml` + 一次次的 run 文件，`state.yaml` 决定哪几个 run 拼成当前视图。

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
  - {id: C1-05, section: C1, ja: "君の声がする", keep_source: true}
```

字段说明：

- `budget`：人工决定的可唱汉字数，结构固定为 `{min, max}`。**MVP 阶段 `min == max`**，退化为定值；将来放开成区间时，数据结构与 C 的比较逻辑都不变。
- `budget: null`：不设约束的乐句（纯语气词、拖长音），C 跳过。
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
- 产物文件只描述结果，不描述血缘（没有 `from_rev` 之类）；血缘在 run 元数据和 `state.yaml` 里。
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

```
data/songs/<song_id>/
  song.yaml                       # L0，人工，agent 只读
  runs/<ts>_<agent>_<hash>/
    request.json                  # 渲染后的入参，调用前就落盘
    output.json
    meta.json                     # model, prompt_hash, input_hash, ts
  preview/                        # --dry-run 的产物
state.yaml

prompts/
  shared.md                       # A / B 共用的总提示词
  a.md
  b.md

corpus/                           # 预留：A 的华语歌词参考语料
evals/                            # 预留：真人评分
```

不用"一首歌一个大 JSON"：人工标注与 agent 产物混写会带来覆盖风险和 diff 噪声，也无法并行跑多个 run 对比。

### 8.1 `state.yaml`

```yaml
song_level:
  analysis: 20261027T075359_a_088264      # A 整曲一次
sections:
  V1: {round: 1, draft: …_b_1a2b3c, review: …_c_4d5e6f, verdict: pass}
  C1: {round: 2, draft: …_b_7g8h9i, review: …_c_0j1k2l, verdict: exhausted}
```

`verdict` 取 `pass` / `fail` / `exhausted` / `skipped`。

指针只决定"当前视图由哪几个 run 拼成"，回滚就是改指针。

### 8.2 LangGraph 的 state

只放指针与小对象：`song_id`、`section_id`（当前 focus）、`round`、三个 run id。大块文本留在磁盘上，checkpointer 不会膨胀，人工介入也只需要改 `focus` 重跑某一段。

## 9. MVP 范围

做：

- `song.yaml` + A 整曲输出 + B 段级生成 + C 的两条规则（`OVER_BUDGET` / `UNDER_BUDGET`）+ `state.yaml`
- `keep_source`（含在 MVP 里，因为它会影响 B 的输入契约，事后加要改契约结构）

暂不做（加进去不改变结构）：

- `budget` 放开成真正的区间（MVP 里 `min == max`）
- 跨行借还字数（`borrow`）
- 行内停顿切分（`pattern`）——句就是乐句，不需要再切
- A 的语料检索（`references`）
- 原文 ↔ 译文的行内词对齐
