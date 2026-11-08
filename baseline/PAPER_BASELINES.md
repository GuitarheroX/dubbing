# 三个 baseline：one-shot 与论文 Exp.4 两组

三个对照都走**同一个入口** [run_baselines.py](run_baselines.py)，输出组织与 workflow 完全一致，只是换提示词和架构。

| 名字 | 生成提示 | 模型实际收到的数据 | 架构 |
| --- | --- | --- | --- |
| `one-shot` | workflow 的 `prompts/shared.md` + `prompts/writer.md` | 整首歌的每个乐句（`id` / `ja` / `budget` / `pattern` / `keep_source`）、**已经定稿的其他段译词**、`focus`；`analysis` 传空 | 按段推进、**只调一次 writer**；没有 perceiver 的意境分析，没有 reviewer 的检查与返修 |
| `paper-original` | 论文 §3 Exp.4，`target language` 替换为 Chinese | 当前段落的完整原歌词，一行一句；不传预算、分组、保留标记、标题或 perceiver 分析 | 每段「生成 + 同模型清理格式」，不返修 |
| `paper-annotated` | Exp.4 的人工预算适配版 | 当前段落的 `id`、`ja`、完整 `budget.min/max`、可选 `pattern` 与 `keep_source` | 同上 |

`one-shot` 是 workflow 的消融对照：workflow 在信息上给它的（整首歌、已定稿的部分、人工预算与分组）它都有，唯一的差别是**没有 workflow 架构本身**——不跑 perceiver 的体会，也没有「reviewer 检查 → 带 prev_zh 返修」的闭环。这样两边的差距才指向架构而不是信息量。

两篇论文组保留按原文 mora / syllable 对齐、押韵与主题的要求；标注组把计数目标换成人工作者给出的预算区间，并说明 `pattern` 只是参考、`keep_source` 的句子原样输出。

## 用法

在项目根目录执行，先预览：

```bash
python baseline/run_baselines.py --song kotozute --baseline all --dry-run
python baseline/run_baselines.py --all --baseline paper --dry-run
```

确认请求后真实运行（需要当前 `.env` 的 API Key）：

```bash
# 三组都用同一份模型、参数与歌曲
python baseline/run_baselines.py --all --baseline all --client api

# 只跑某一种或某一首
python baseline/run_baselines.py --song kotozute --baseline one-shot --client api
python baseline/run_baselines.py --all --baseline paper-original --client api
```

`--baseline` 可重复，取值 `one-shot` / `paper-original` / `paper-annotated` / `paper`（= 两组论文）/ `all`（默认）。`--song` 可重复；`--root` 可指定歌库；`--model`、`--base-url`、`--temperature`、`--max-tokens` 可覆盖当前配置。

默认 `--protocol anthropic`，模型、接口地址和密钥从 `.env` / 进程环境读取。可用 `--jobs 2` 并发跑不同歌曲 / 组别；每组任务内部仍按顺序调用。OpenAI 兼容网关可另用 `--protocol openai`，此时调用 `/chat/completions` 并使用 Bearer 认证。

默认 `--thinking none`：**不发送思考开关**，三组条件一致。需要复现此前关闭思考的那一批时加 `--thinking disabled`；OpenAI 协议会发 `enable_thinking: false`，Anthropic 协议发 `thinking: {type: disabled}`。若请求关闭思考但网关仍返回推理内容或正数 reasoning tokens，该任务会报错。

每段生成与格式清理使用同一设置；`one-shot` 每段只调一次。实际请求体与设置逐条存在该次调用的 `request.json` / `meta.json` 里。

## 结果目录

一次运行一个 run 目录，旧的绝不覆盖：

```text
data/songs/<song_id>/baselines/<名字>/
  latest.json                       # 指向当前 run
  run-<UTC 时间戳>-<8 位随机>/
    sections/001/writer/{request,output,meta}.json          # one-shot
    sections/001/generation/{request,output,meta}.json # paper-*
    sections/001/cleanup/{request,output,meta}.json    # paper-*
    sections/002/...
    lyrics.zh.txt
    lyrics.md
```

- `request.json`：这次调用的完整入参。
- `output.json`：解析后的产物。paper 组的 `generation/output.json` 是逐句译文（另记 `source`，说明是按生成里的结构化 JSON 取的还是按清理行对齐的），`cleanup/output.json` 是清理出来的行数组。
- `meta.json`：模型、接口、温度、用量、提示词 hash；模型原始正文存在 `raw_response` 里。paper 组不再另存 `config.json` / `song.json` / `report.json` / `response.txt`，论文对应关系（`paper`、`experiment`、`fidelity`）折进 `meta.json`。

paper 组每段调用两次：一次生成，一次由同模型清理输出格式。清理提示两组完全相同，只接收该次生成结果，不接收人工预算、原文或规则反馈。清理后行数必须恰好一行对应一个输入乐句；**行数不符就记录失败**，不凭行数猜测并删除首行，也不补写歌词。

若生成响应本身已是含准确、唯一 phrase ID 和 `zh` 的 JSON，程序优先按 ID 直接提取原始 `zh`，而不采用清理模型可能误删或改写的内容；两组使用相同规则，来源记在 `generation/output.json` 的 `source` 字段。未知、缺失、重复 ID 会报错。

## 评测

`lyrics.md` 的「状态」列由确定性的规则检查（`dubbing/checks.py`）在生成结束后**事后**算出，只用于导出口径；规则结论不进入任何模型请求，也**不**单独落盘成报告文件。加入人工预算、字数距离、跨组比较等评测是后续任务。

## 与论文的对应关系与出处

`paper_original.txt` 录入论文 §3 Exp.4 的指令，正文合并排版换行，唯一模板替换为目标语言；`paper_annotated.txt` 明确修改计数要求并增加人工输入格式说明，不能称作论文未改动提示词。两者的押韵和主题要求均保留。

出处：Hanze Liu, Yusuke Sakai, Taro Watanabe. 2026. *Towards Singable Lyrics Translation Using Large Language Models*, EACL Student Research Workshop, pp.544–554，§3，p.546。[论文页](https://aclanthology.org/2026.eacl-srw.42/)、[PDF](https://aclanthology.org/2026.eacl-srw.42.pdf)。指令按 [ACL 2016 年及以后材料的 CC BY 4.0 授权说明](https://aclanthology.org/faq/copyright/) 转载并注明适配；不复制论文示例中的第三方歌词。

**“原样”指生成提示与无人工标注输入，不代表整套论文实验的严格复现。** 按本项目选择，三组共用当前模型，不切换到论文的 GPT-4o；也不强制论文的 `temperature=0, top_p=0`。记录的是实际请求参数，不保证服务端有效采样参数或完全确定性。DeepSeek 兼容接口可能固定或限制采样参数，见[接口说明](https://api-docs.deepseek.com/guides/anthropic_api/)。

论文说明另有清洗 API 与人工核对，但未给出完整清洗提示、消息包装及所有生成设置。本实现的消息角色和格式清理提示是显式留档的本地重建，不能自动宣称复现论文得分。详细前期核对见[研究笔记](../docs/paper-baseline-research.md)。

## 从旧布局迁移

旧的 `runs/` + `state.yaml` 与论文组的 `config.json` / `report.json` 等，早已由一次性的 `scripts/migrate_layout.py` 整理掉（脚本跑完即删，默认只预览、加 `--apply` 才落盘）。
