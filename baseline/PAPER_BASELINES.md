# 论文 Exp.4 双 baseline

这两组按同一段落划分、使用当前项目配置的同一模型运行，只改变是否提供人工标注。现有 `run_baseline.py`、`prompt.txt` 和各首歌的 `baseline.txt` 保留原有行为。

| 组别 | 生成提示 | 模型实际收到的歌词数据 |
| --- | --- | --- |
| `paper-original` | 论文 §3 Exp.4，`target language` 替换为 Chinese | 当前段落的完整原歌词，一行一句；不传预算、分组、保留标记、标题或 A 分析 |
| `paper-annotated` | Exp.4 的人工预算适配版 | 当前段落的 `id`、`ja`、完整 `budget.min/max`、可选 `pattern` 与 `keep_source` |

原样组保留按原文 mora / syllable 数量对齐、押韵与主题的要求；标注组将计数目标替换为人工预算区间，另外说明 `pattern` 为参考、保留句需原样输出。两组都按 section 独立调用，不是 Exp.7 的整曲共享会话，也不使用 A 分析或 C 驱动返修。

## 用法

在项目根目录执行，先预览：

```bash
python baseline/run_paper_baselines.py --song kotozute --dry-run
python baseline/run_paper_baselines.py --all --variant both --dry-run
```

确认请求后真实运行（需要当前 `.env` 的 API Key）：

```bash
# 两组使用相同的当前模型、参数和歌曲
python baseline/run_paper_baselines.py --all --variant both --client api

# 也可先分别跑一首
python baseline/run_paper_baselines.py --song kotozute --variant paper-original --client api
python baseline/run_paper_baselines.py --song kotozute --variant paper-annotated --client api
```

`--song` 可重复；`--root` 可指定歌库；`--model`、`--base-url`、`--temperature`、`--max-tokens` 可覆盖当前配置。默认读取项目的 `DUBBING_MODEL`、`DUBBING_BASE_URL`、通用 `DUBBING_TEMPERATURE`；两组均用 `baseline` 角色，不应用 A / B 专用温度。没有 `--client api` 时不会产生真实调用。

默认 `--protocol anthropic` 沿用项目原接口，模型、接口地址和密钥从 `.env` / 进程环境读取。当前重跑使用 `.env` 中的 `deepseek-flash`、`https://api.deepseek.com/anthropic` 和通用温度 `0.7`，并显式传入 `--thinking disabled`。可用 `--jobs 2` 并发运行两个独立歌曲 / 组别任务；生成与清理仍在各任务内按顺序执行，默认并发数为 1。OpenAI 兼容网关可另用 `--protocol openai`，此时调用 `/chat/completions` 并使用 Bearer 认证。

默认 `--thinking disabled`。OpenAI 兼容网关的请求体显式发送 `enable_thinking: false`；Anthropic 协议则发送 `thinking: {type: disabled}`。生成和格式清理使用同一设置。实际请求体（不含认证头）存入 `request.json`，设置存入 `config.json`，响应中的推理统计存入阶段元数据；若请求关闭思考但网关仍返回推理内容或正数 reasoning tokens，该任务会报错。

网络或网关暂时失败后，可在同一命令后加 `--resume --jobs 1 --max-retries 8`。这会检查各组最新 run 的数据、提示词、模型和生成设置；一致时跳过已完成结果、复用已有成功调用，只重试失败或尚未执行的步骤。失败调用的旧记录放入该阶段的 `attempts/`，根元数据保留续跑历史；已生成的译词不会为提高分数而重新抽样。设置不同会拒绝续跑，需去掉 `--resume` 新建运行。行数等格式失败只重用原输出，无法修复时仍报告失败。

每组每个段落调用两次：一次生成，一次由同模型清理输出格式。清理提示两组完全相同，只接收该次生成结果，不接收人工预算、原文或规则反馈。清理后必须恰好一行对应一个输入乐句；行数不符就记录失败，不凭行数猜测并删除首行，也不补写歌词。清理模型仍可能改动文字，需要人工核对原始响应与清理结果。

若生成响应本身已是含准确、唯一 phrase ID 和 `zh` 的 JSON，程序优先按 ID 直接提取原始 `zh`，而不采用清理模型可能误删或改写的内容；这不会新增模型调用或修改译词。两组使用相同提取规则，并在 `alignment.json` 记录来源。未知、缺失、重复 ID 会报错。

## 结果与评测

每次运行生成唯一目录，预览同样独立，不覆盖旧结果：

```text
data/songs/<song_id>/baselines/<variant>/<run_id>/
  config.json          # 模型、请求参数、提示词/源码/数据哈希、复现差异
  song.json            # 完整数据快照，供事后对齐与统一评测；不等于模型输入
  meta.json            # 完成或失败状态、各成功调用用量
  sections/001/
    section.json       # 原始 section ID、phrase ID 映射
    generation/
      request.json     # 实际提示词、原文或标注输入
      response.txt     # 模型正文
      response.json    # 若取得，完整 JSON 响应（包括 stop_reason / usage）
      meta.json
    cleanup/           # 清理调用的同类记录
    output.json        # 通过行数检查的逐句结果
  output.json          # 全曲逐句结果，含模型译文与最终演唱文本
  report.json          # 两组统一的人工预算与规则评测
  lyrics.zh.txt
  lyrics.md
```

原样组也会翻译项目标为 `keep_source` 的句子，因为生成时不传这个人工标记。两组事后导出统一按项目标记还原这些句子的原文，并从预算与规则通过率中排除。模型实际译出的内容仍保存在 `model_text` 和逐段输出中。原样组的 `song.json` 与 `report.json` 可以含人工预算用于本地评测，但 `generation/request.json` 的 system/user 中没有这些标注。

规则检查只评分不返修，预算未通过的生成也会完整保存。`meta.json` 的 `completed` 表示生成、格式对齐和导出完成，不代表预算全部通过。`report.json` 分别给出预算通过率、平均预算距离、规则通过率和逐句诊断。预算距离为低于下限或高于上限的差值，区间内为 0；保留句和无预算句不进入预算指标。这些指标不等于论文原始 source-count 节奏距离，也不能代替自然度、情绪和真人贴曲评测。

这些新运行目录默认被 Git 忽略。需要提交快照时应检查内容后按路径 `git add -f`，不要提交 `.env`。既有主流水线的 `state.yaml`、`runs/` 和译词文件均不改动。

## 与论文的对应关系与出处

`paper_original.txt` 录入论文 §3 Exp.4 的指令，正文合并排版换行，唯一模板替换为目标语言；`paper_annotated.txt` 明确修改计数要求并增加人工输入格式说明，不能称作论文未改动提示词。两者的押韵和主题要求均保留。

出处：Hanze Liu, Yusuke Sakai, Taro Watanabe. 2026. *Towards Singable Lyrics Translation Using Large Language Models*, EACL Student Research Workshop, pp.544–554，§3，p.546。[论文页](https://aclanthology.org/2026.eacl-srw.42/)、[PDF](https://aclanthology.org/2026.eacl-srw.42.pdf)。指令按 [ACL 2016 年及以后材料的 CC BY 4.0 授权说明](https://aclanthology.org/faq/copyright/) 转载并注明适配；不复制论文示例中的第三方歌词。

**“原样”指生成提示与无人工标注输入，不代表整套论文实验的严格复现。** 按本项目选择，两组共用当前模型，不切换到论文的 GPT-4o；也不强制论文的 `temperature=0, top_p=0`。记录的是实际请求参数，不保证服务端有效采样参数或完全确定性。DeepSeek 兼容接口可能固定或限制采样参数，见[接口说明](https://api-docs.deepseek.com/guides/anthropic_api/)。

论文说明另有清洗 API 与人工核对，但未给出完整清洗提示、消息包装及所有生成设置。本实现的消息角色和格式清理提示是显式留档的本地重建，运行后 `manual_verification` 为 `pending`，不能自动宣称复现论文得分。详细前期核对见 [研究笔记](../docs/paper-baseline-research.md)。
