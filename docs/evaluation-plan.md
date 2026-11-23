# 四组译配结果评测方案

更新时间：2026-11-08。本阶段比较四组结果，只将音节约束、文本质量、相对质量和 COMET-QE 作为主评价项。流程已实现于 `evaluate/`，本次仅调整代码、文档和测试，尚未启动正式评测或调用评审 API，也未下载 COMET 权重。

## 1. 比较对象与数据来源

| 组别 | 处理方式 | 读取的最终结果 |
| --- | --- | --- |
| `paper-original` | EACL 论文 Exp.4 原提示词；生成时只提供本段原文 | `baselines/paper-original/<run>/sections/NNN/generation/output.json` |
| `paper-annotated` | Exp.4 人工标注适配版，提供 budget、可选 pattern、keep_source | `baselines/paper-annotated/<run>/sections/NNN/generation/output.json` |
| `workflow` | 不加入 RAG 的 perceiver → writer → reviewer 流程 | `workflow/<run>/sections/NNN/writer/output.json` |
| `workflow-rag` | 加入检索与改写的 workflow-RAG | `workflow_rag/<run>/sections/NNN/mimic_write/output.json` |

`paper-annotated` 不只是添加 pattern：计数目标也从源文 mora/syllable 改为人工 budget，输入格式变为 JSON。pattern 仍为参考分组，不自动成为硬约束。两条 workflow 都使用各自的最终结果，不能把 RAG 的 writer 初稿当成最终改写结果。`one-shot` 暂不纳入四组比较。

当前默认优先使用包含歌曲的 `data/songs_rag/`，因为它同时存有两组论文 baseline、workflow 和 workflow_rag；否则回退到 `data/songs/`。可用 `--root` 显式指定共同歌库，也可用 `--baseline-root`、`--workflow-root`、`--rag-root` 分别指定来源。

跨目录比较必须确认原文、分句、预算、pattern、保留标记一致。程序对此进行校验；来源不同但标注一致可以比较，标注不同会报错。程序兼容旧 baseline 快照、旧 state.yaml/runs，以及已迁移但仍叫 A/B/C 的目录，不改写原始结果。

新版原样 baseline 按设计没有收到人工标注，也没有单独保存完整 song.json；只能核实其原文与对齐 ID，不能声称验证了生成时的人工预算版本。这个限制会保存在来源信息中。

## 2. 本阶段四类指标

| 类别 | 指标 | 实现方式 | 输出位置 |
| --- | --- | --- | --- |
| 音节约束 | 预算通过率、平均预算距离、偏短/偏长比例 | 复用计数器和 C，在相同人工预算下重新计算四组结果 | rules.json、per-song.json、summary.json |
| 文本质量 | 核心语义、情绪意境、自然度、表现力 | LLM 对匿名候选各项独立给 1–5 分，提供简短理由和乐句 ID | judges/、summary.json |
| 相对质量 | 成对偏好胜率及顺序分歧 | 四组共 6 对，每对交换顺序评两次，允许平局或弃评 | judges/、summary.json |
| 语义诊断 | COMET-QE | 日文原文 + 候选中文译词，本地运行固定评测模型 | comet.json、summary.json |

四类指标分别报告，不合成一个加权总分。完成率、覆盖率、错误分布用于说明有效样本，不是新增的主质量指标。pattern 显式分组检查和硬规则信息可以作为诊断保留，不用它们替代实际可唱性。

实际旋律贴合度与跟唱难度不在当前四项内，纯文本模型不为其评分。已有真人记录导入功能保留为后续扩展，导入后也仅在附录单列。

## 3. 音节约束的计算

第 i 句的实际计数为 n_i，人工预算为 [lo_i, hi_i]：

```text
pass_i = 1，若 lo_i <= n_i <= hi_i；否则为 0
距离_i = max(lo_i - n_i, n_i - hi_i, 0)

预算通过率 = sum(pass_i) / 有效乐句数
平均预算距离 = sum(距离_i) / 有效乐句数
偏短比例 = count(n_i < lo_i) / 有效乐句数
偏长比例 = count(n_i > hi_i) / 有效乐句数
```

例如预算 [8, 10]，实际 7、9、12 音节时，距离为 1、0、2。复用 [count_syllables](../dubbing/counting.py)、[C 检查](../dubbing/checks.py)，由 [rules.py](../evaluate/rules.py) 统一计算。汉字、外文和儿化等计数局限随报告说明，人工预算自身也需要演唱校准。

保留原文句不参加中文预算与文本质量评分；无预算句可参加文本质量评分和覆盖率统计，但不进入预算指标。缺失或无法对齐的译词标为 missing，不按空串记入长度指标，也不给模型质量填零。导出占位符不能算有效译词。

原样组要求匹配源文 mora/syllable，其目标与人工预算不同。这里统一衡量本项目的预算目标，不能将它称为原论文的节奏距离，也不能仅凭低预算分宣布论文策略失效。

## 4. 文本质量：LLM 匿名评审

以 section 为单位，向评审提供相同原文上下文、当前段落及四份匿名候选。评分候选使用 A/B/C/D 标签、随机排序；真实分组映射仅在本地存档，不发送给模型。模型不接收方法名、生成模型、A/perceiver 分析、C/reviewer 结果或已有分数。

| 维度 | 评价内容 | 1 / 3 / 5 分锚点 |
| --- | --- | --- |
| semantic 核心语义 | 主题、人物关系、事件、叙述视角 | 核心误读 / 主旨保留但明显偏移 / 核心信息准确且改写合理 |
| emotion 情绪意境 | 情绪方向、强弱、氛围与段落变化 | 相反或无关 / 基本一致但平淡或局部偏离 / 情绪与意境充分保留 |
| naturalness 自然度 | 是否符合中文歌词文体，是否生硬凑字 | 明显不自然 / 可理解但多处生硬 / 自然顺口 |
| expressiveness 表现力 | 意象、重复和表达是否有效且服务主题 | 空泛混乱 / 有表达但普通 / 鲜明、有感染力且贴合主题 |

2、4 分介于相邻锚点之间。允许合理的艺术改写，不因更长、更华丽或意象更多自动加分；不按散文标准惩罚歌词片段、省略和重复。语义、情绪、自然度和表现力独立评价，不因字数达标自动给高分。

每项须返回整数 1–5 分、短理由和合法的 phrase ID；无法判断时返回 null 及原因。非法评分或缺少依据会记录为失败，不凭空补分。评分说明见 [score.txt](../evaluate/prompts/score.txt)。

各维度的主比较仅使用四组都得到有效评分的同一批段落，再先按歌平均、最后按歌曲等权平均；单组可用评分与覆盖率同时保留。优先选择不同于生成模型的评审模型，固定提示词和参数，记录版本与响应。有条件可通过第二个独立模型或少量真人样本检查稳定性，但不能把模型一致视为已经验证符合人类感受。

## 5. 相对质量：六对比较与顺序复评

四组形成以下 6 对：

1. 原提示词 / 人工标注。
2. 原提示词 / 无 RAG workflow。
3. 原提示词 / workflow-RAG。
4. 人工标注 / 无 RAG workflow。
5. 人工标注 / workflow-RAG。
6. 无 RAG workflow / workflow-RAG。

每对分别按 A/B 和 B/A 展示两次，允许选 A、B、tie 或 null。映射回真实分组后，两次结论一致才计入胜率；冲突记 unstable，无法判断记 abstained，任务失败记 missing，分别报告数量。

```text
每歌偏好胜率 =（胜出次数 + 0.5 × 平局次数）/ 该歌顺序一致的有效比较数
总体偏好胜率 = 有效歌曲的每歌偏好胜率的平均值
```

同时保存按有效比较数直接汇总的微平均。没有有效比较时为 null，不填 0。稳定率、不稳定率、弃评和缺失必须随胜率展示，避免只看剔除不稳定样本后的胜率。

RAG 作用重点读取第 6 对，同时保留其他 5 对。报告单列 workflow-rag 相对 workflow 的胜率和规则指标差值。每个完整段落共计划 **13 次模型调用：1 次四候选评分 + 6 对 × 2 次顺序评审**。可先 dry-run 查看请求和任务数，正式调用仍需显式启用。

## 6. COMET-QE

默认可选模型为论文所用的 `Unbabel/wmt22-cometkiwi-da`，按句输入 src=日文原文、mt=候选中文译词，不提供参考译文、人工预算或旋律。保存句级分数，再进行逐歌宏平均；主比较采用四组共同完成歌曲。

记录实际解析的模型 revision、checkpoint 和 hparams 哈希、编码器可获得的版本信息、依赖版本、推理配置。只有显式 `--comet` 才导入并运行模型；未启用时不下载权重。模型缓存或本地权重可通过 `--comet-local-only` 使用。

这是通用翻译质量诊断，可能对自由填词中的语义变化扣分，不代替情绪或真实可唱性。现有系统输出不能作为金标准参考。原始分数不是准确率，也不与 1–5 的模型评分直接相加。

## 7. 一致性与统计

- 以 (song_directory, section, phrase_id, variant) 对齐，目录名与 song.id 分别保留。
- 从各组 latest 或指定批次选择 run 后，冻结输入、来源路径及哈希；续跑不跟随后续 latest 变化。
- 四组比较采用共同完成的歌曲；各项有效分母和全库覆盖率另报，不把某组缺失样本默默计入零分或隐藏。
- 先逐歌计算、再按歌曲等权平均，句级微平均作为补充。规则差值按歌曲做配对 bootstrap，默认 2,000 次、seed=0，报告 95% 区间；它只反映歌曲抽样，不涵盖全部模型采样波动。
- 元数据记录各组生成模型、提示词 hash、温度、思考开关和角色。历史记录缺少某个设置时标为未知，不推定相同。
- 只有在生成模型、共享提示词、采样配置和数据版本可比时，workflow 与 workflow-rag 的差值才更接近 RAG 的贡献；现有产物比较本身不能自动构成严格消融实验。
- 历史三组比较报告保留为历史记录，不能直接冒充当前四组评测结果。原有三组评测目录不用于新四组任务续跑。

## 8. 实现与产物

| 模块 | 职责 |
| --- | --- |
| [inputs.py](../evaluate/inputs.py) | 读取四组及新旧布局、校验跨目录版本、冻结输入 |
| [rules.py](../evaluate/rules.py) | 预算指标、覆盖诊断、逐歌配对差值与区间 |
| [judges.py](../evaluate/judges.py) | 匿名四候选评分、六对顺序复评、失败记录、缓存与续跑 |
| [comet_qe.py](../evaluate/comet_qe.py) | 可选本地 COMET-QE，记录固定模型信息 |
| [report.py](../evaluate/report.py) | 四组、四类指标的 JSON 和 Markdown 报告 |

```text
data/evaluations/<evaluation_id>/
  manifest.json       # 代码、输入哈希、种子与统计设置
  inputs.json         # 四组固定来源、候选译词、标注与缺失状态
  rules.json          # 逐句约束与完整性诊断
  judges/             # 匿名映射、请求、响应、理由、分数与顺序分歧
  comet.json          # 可选本地语义诊断
  per-song.json       # 每歌指标及有效分母
  summary.json        # 四类主指标、六对比较和 RAG 对照差值
  report.md           # 可读报告，不捏造未运行阶段的分数
```

默认运行仅计算离线约束；`--judge-dry-run` 仅生成匿名请求，`--judge --client api` 才调用模型。凭据从环境读取，不写入产物。失败评审续跑只补失败任务，不重新生成译词。完整命令见 [evaluate/README.md](../evaluate/README.md)。

测试使用临时合成数据、mock API 和假 COMET 后端，覆盖四组隔离、六对反转、缺失分母、跨根目录标注不一致、无效评分、缓存配置变化、凭据不落盘及完整流程。实现和测试不代表正式评测已开始。

## 9. 依据

- [Towards Singable Lyrics Translation Using Large Language Models](https://aclanthology.org/2026.eacl-srw.42/)：Exp.4 提示词与 §4.3 的 COMET-QE；本项目仍需明确模型、清理及任务目标的差异。
- [COMET 官方实现](https://github.com/Unbabel/COMET#reference-free-evaluation)：无参考式翻译质量估计的输入与运行方式。
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)、[G-Eval](https://aclanthology.org/2023.emnlp-main.153/)：模型评审方法及顺序、风格等偏差，不直接证明歌词领域的人类一致性。
