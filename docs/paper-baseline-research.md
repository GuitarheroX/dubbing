# 可唱歌词翻译论文 baseline 核对

核对日期：2026-11-05。范围：论文全文（11 页）及作者官方页面；未改程序，未调用付费模型 API。

## 论文事实速查

Strategies (§3): 1 direct; 2 line alignment; 3 rhythm/rhyme/theme constraints; 4 explicit verification; 5 two-turn general refinement; 6 two-turn rhythm refinement; 7 sequential song sections sharing conversation history. Prompts appear in the main text, with no appendix. Setup (§4.2): gpt-4o-2024-11-20; temperature=0, top_p=0; section inputs; separate cleanup call and manual verification. Corpus (§4.1): 96 songs, 624 sections, six directions. Metrics (§4.3): SBERT, COMET, COMET-QE, rhythm distance, rhyme-edge Jaccard, CCVO distance. Release statement (pp.552–553): prompts/code/derived annotations, excluding full lyrics/audio. [论文 PDF](https://aclanthology.org/2026.eacl-srw.42.pdf)

为避免复制大段原文，七项英文全名及指令原文直接定位至 PDF pp.546–547、§3 各实验框。这里的序号始终对应论文 Exp.1–7。

## 官方材料公开状态

本次未找到可核实的作者代码仓库或派生标注下载入口，不能把论文的公开声明当作已经取得代码/数据。

| 已核查的一手入口 | 本次可见内容 |
| --- | --- |
| [ACL Anthology 论文页](https://aclanthology.org/2026.eacl-srw.42/) | PDF、引用与元数据；未见代码或数据附件 |
| [NAIST 官方论文列表](https://nlp.naist.jp/en/publications/) | Workshops 内有论文条目，未附实现/数据链接 |
| [NAIST 官方软件页](https://nlp.naist.jp/en/software/) | 列出其他软件，未见本论文项目 |
| [共同作者 Yusuke Sakai 官网](https://www.yusuke1997.jp/) | 2026 年第 31 项只有 paper 链接 |

另检索过精确标题加 GitHub、论文通讯邮箱；没有找到可可靠归属本论文的仓库。这是本次检索结果，不代表作者从未公开过材料。

## 复现前待确认

以下问题应通过作者材料或联系作者解决；当前不应自行填成“论文配置”。

- 日→中具体歌曲、段落及参考译文清单是什么？不能从总量直接推导该方向的样本数，也不能把本地 20 首称作论文测试集。
- §3 的指令框之外，完整 API 消息角色、源歌词包装和清洗提示词是什么？每个生成阶段的 token 上限、停止条件及服务端配置如何设置？
- 音系转写、日文读音消歧、中文多音字/儿化、押韵类别、空押韵集合、空行和不等行数如何处理？参考与源文本的使用位置、聚合方法须和实现逐项核对。
- §4.3 与 §7.3 的 CCVO 定义、相关性符号是否使用同一个量？复现时应分别保存原始距离和任何变换后的分数，并检查方向，不能照抄结论来决定实现。
- Table 2 是跨方向聚合；需要日→中逐策略结果和不确定性，才能判断该方向的优胜策略。当前不能把聚合人评结论推广成日→中排名。

## 本项目适用方案（前期建议）

建议保留现有 baseline 为 legacy，另建 paper 策略入口和独立结果目录。主 baseline 选择 Exp.4；Exp.2 做简单对照，Exp.7 做保留整曲上下文的对照，Exp.6 作为可选的两轮节奏修订。选择 Exp.4 是实验成本与约束明确程度的取舍，不是宣称它已被证明最适合本项目。

应区分两组实验：`paper-original` 尽量保留原论文的计数目标和指令；`paper-adapted-budget` 仅先把计数目标换为本项目人工 budget。后续任何提示词、输入字段或清洗变化分别留记录。不要给这些 baseline 接入 perceiver 或 reviewer 的额外修订，否则无法识别 prompting 策略本身的效果。另设 writer-only、去 perceiver、去 reviewer 的消融，判断现有流水线的收益来源。

根据主线程本地审阅，当时有 20 首、421 句，其中 3 句保留原文、418 句可译；10 首已有 `baseline.txt`。当时那个 `baseline/run_baseline.py`（后来已被 `one-shot` 取代，见文末）按全曲一次调用，省略无 budget 句、把区间预算取上界，并在输出多一行时删除首行。它需要增加可靠的逐句对齐和原始输出留存；不能沿用该首行删除规则承担论文实验的清洗。当前 API 封装还需要支持任意消息历史和显式 `top_p`，才能实现逐段同对话实验。

建议先选 3 首差异较大的歌曲，只生成离线请求预览，核对段落切分、轮次、歌词 ID、保留原文句以及清洗输入；确认后再开展付费实验。全部策略应使用同一模型、解码参数、输出预算和数据快照。按歌保存请求、每轮原始输出、清洗结果、usage、配置及代码版本；额外清洗调用的费用也计入成本。

评测分别报告人工预算通过率、预算距离、源文计数的节奏距离；三个量不能合并。保留句级诊断，同时按歌宏平均，并用按歌曲配对的 bootstrap 表示策略差值的不确定性。没有独立人工参考时，不把本项目生成译词当成金标准。盲评覆盖自然度、贴曲程度、情绪与总体质量；语义评估与可唱性评估分别呈现。

## 已落地的首阶段

按后续确定的范围，先实现 Exp.4 两组：保留原生成提示的 `paper-original`，以及传入人工 budget / pattern 的 `paper-annotated`；两组都使用当前项目模型。三组 baseline 现在合并到统一入口 [run_baselines.py](../baseline/run_baselines.py)，输出组织与 workflow 一致（`baselines/<名字>/<run_id>/sections/NNN/...`），详细对应关系与差异见 [baseline 说明](../baseline/PAPER_BASELINES.md)。

原先那个「整曲一次调用」的 legacy baseline 已被 `one-shot` 取代：它改用 workflow 的提示词（`prompts/shared.md` + `prompts/writer.md`）、按段推进并带上已经定稿的其他段，只是不跑 perceiver 的意境分析与 reviewer 的检查返修，从而把「架构」单独隔离出来。原样指生成提示，不表示模型、解码参数、消息包装和清洗环节与论文完全一致。其他策略、原论文音系指标与人评流程尚未实现。
