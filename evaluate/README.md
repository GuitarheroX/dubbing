# 结果评测与基线比较

默认比较四组结果：`paper-original`（EACL Exp.4 原提示词）、`paper-annotated`（人工 budget / pattern）、`workflow`（无 RAG）、`workflow-rag`（加入 RAG）。可通过 `--variant` 加入 `one-shot` 消融或选择方法子集。只读取已有结果，不重跑生成流程。

本阶段聚焦四类指标：音节约束、LLM 文本质量、成对偏好、COMET-QE。真实调用需显式选择 `--judge --client api`；COMET-QE 单独启用。

```text
读取并冻结四组输入 → 离线规则统计 → 可选 LLM 匿名评审 → 可选本地 COMET-QE
```

默认只运行离线约束统计，不调用付费 API 或下载 COMET 权重。完成率/覆盖率是有效样本诊断，不作为额外质量分；实际可唱性不在当前四项指标中，纯文本模型不为它补分。

## 1. 离线统计

在项目根目录执行：

```bash
python -m evaluate --root data/songs_rag
```

默认优先选取包含歌曲的 `data/songs_rag/`；否则使用 `data/songs/`。省略 `--batch` 时冻结两组 baseline 的 latest run。workflow 读取 `workflow/` 的 writer，workflow-rag 读取 `workflow_rag/` 的 mimic_write，不把 writer 初稿当作 RAG 结果。兼容旧 A/B/C 目录名和旧 state.yaml/runs，读取不修改原文件。

各组默认共享同一歌库。需要分别取数时，可指定 `--baseline-root`（含 one-shot）、`--workflow-root`、`--rag-root`；程序要求各目录的原文、分句、budget、pattern 和保留标记一致，否则报错，不静默混用。源路径、模型和生成设置均留档；条件不一致时不能把差值仅归因为 RAG。

加入单次 writer 消融的五组比较：

```bash
# 先生成 one-shot；每段只调用一次 writer，不提供 perceiver 分析或 reviewer 返修
python baseline/run_baselines.py --root data/songs_rag --all --baseline one-shot \
  --client api --thinking disabled --temperature 0.7 --jobs 3

# 冻结五组并预览评审；不传 --variant 时仍为原有四组
python -m evaluate --root data/songs_rag \
  --variant paper-original --variant paper-annotated --variant workflow \
  --variant workflow-rag --variant one-shot \
  --out data/evaluations/five-way --judge-dry-run
```

只比较消融和完整流程可用 `--variant workflow --variant one-shot`。one-shot 从 writer（兼容旧 B）记录读取，核对全曲输入、focus 和逐句输出；不把事后规则检查作为模型返修。

对比历史 workflow 时还应核对实际系统提示词、温度和思考设置。当前 writer 将精确字数对齐交给后续角色，而部分历史 writer 提示要求自身严格对齐；这样的产物比较不能称为只改变架构的严格消融。

`one-shot-archive` 单独读取 `baselines/one-shot-archive/`，使用配套 `archive/shared.md` + `archive/b.md`，保留历史 writer 的严格预算要求。与五组同时比较时，显式列出六个 `--variant`：

```bash
python -m evaluate --root data/songs_rag \
  --variant paper-original --variant paper-annotated --variant workflow \
  --variant workflow-rag --variant one-shot --variant one-shot-archive \
  --out data/evaluations/six-way --judge-dry-run
```

六组每段计划 31 个任务（一次六候选打分，15 对候选各交换顺序复评）。可复用五组目录中请求和配置一致的成对任务；六候选的绝对打分需重新调用。

```bash
# 只评选定歌曲（使用目录名，--song 可以重复）
python -m evaluate --song abuku --song bara

# 采用历史固定 baseline 批次，其余两组取 songs_rag
python -m evaluate --root data/songs_rag --baseline-root data/songs \
  --batch data/baseline_batches/20261105T124026Z/final-summary.json

# 新建指定输出目录，已有目录会被拒绝，避免覆盖
python -m evaluate --out data/evaluations/my-comparison
```

产物默认写入 `data/evaluations/eval-<timestamp>-<id>/`：

```text
manifest.json        # 输入和代码哈希、随机种子、bootstrap 设置
inputs.json          # 固定的歌词、预算、候选译词、缺失状态、源 run
rules.json           # 逐句检查、全库覆盖、共同完成歌曲的配对统计
per-song.json        # 逐歌汇总，包含有效分母
summary.json         # 各阶段汇总；未运行的阶段保持未评测
report.md            # 可读对比报告
human-template.json  # 真人记录模板，不自动生成真人分数
```

音节指标复用项目计数器和 C。保留句排除；无预算句参与覆盖率、不参与预算分。缺失译词不按空串计数，也不填成模型质量 0 分。完成率和覆盖率按全库报告，主对比按四组共同完成歌曲做歌等权宏平均，配对 bootstrap 默认 2,000 次、seed=0。

读取新版 baseline 时，从 `generation/request.json` 验证输入、从已对齐的 `generation/output.json` 取译词；`source: null` 的未对齐占位输出不计入覆盖率。原样组未收到人工标注，因此只能核实原文版本，不能声称验证了生成时的人工预算版本。旧版带 `song.json/config.json` 的 baseline 仍可读取。

原样组的计数目标为源文 mora/syllable，人工预算是本项目目标，报告会明确两者差异。硬规则诊断不是独立可唱性验证。

## 2. LLM 文本质量与成对偏好

先预览一个段落的完整请求：

```bash
python -m evaluate --judge-dry-run --judge-limit 1
```

这会输出一个新的评测目录，以下简称 `<评测目录>`。将该占位符替换为实际路径。相同输入及配置可以从预览继续真实调用：

```bash
python -m evaluate --resume <评测目录> --judge --client api --judge-limit 1
```

全库真实评审可在新目录一次运行，不使用 `--judge-limit`：

```bash
python -m evaluate --judge --client api
```

每个四组共同完成且包含需译乐句的 section，计划 **13 次评审调用**：一次对四个匿名候选评分，6 对候选各交换顺序评两次（1 + 6 × 2）。调用前可通过 dry-run 的 `judges/judge-1/summary.json` 查看目标段落数和任务数。全部调用都会单独存档；不会修改候选译词。

五组时每段为 **21 个任务**：一次五候选评分，加 10 对候选各交换顺序复评。`--reuse-judge <旧评测目录>/judges/judge-1` 可复用提示词、模型配置和完整请求一致的已完成任务；程序验证原始响应哈希和解析结果。四组扩为五组时，只能复用原四组的成对评审，绝对打分需包含五候选重新调用，不拼接旧评分。复用来源及任务 ID 留在 `reused-from.json`，旧报告不修改。

参数说明：

| 参数 | 默认 / 用途 |
| --- | --- |
| `--model`、`--base-url` | 从 `.env` / 进程环境读取当前模型和接口 |
| `--protocol` | `anthropic`；可选 `openai` |
| `--temperature` | `0`，不保证完全确定 |
| `--thinking` | `disabled`，响应若仍返回思考内容则记录失败 |
| `--max-tokens` | `6000` |
| `--max-retries` | `3`，用于网络/网关错误重试 |
| `--judge-limit` | 只选择前 N 段；预览和续跑必须一致 |
| `--judge-jobs` | 默认 `1`；可设置为 `4` 并发，每个任务使用独立客户端 |
| `--judge-name` | `judge-1`，第二个评审模型用不同名称保存 |

优先选用不同于生成模型的评审模型。同一生成模型也可以运行，但报告应披露这一限制。凭据不写入请求文件；只保存不含认证头的请求体。

输入包括相同的原文上下文、当前段落及匿名候选。模型看不到方法名、源 run、A 分析、C 结果或已有分数。绝对评分采用 `semantic`、`emotion`、`naturalness`、`expressiveness` 四项 1–5 分，每项须有短理由及合法 phrase ID；材料不足可返回 null。评分锚点见 [score.txt](prompts/score.txt)。

成对评审允许 A、B、tie、null。程序复原候选身份，交换顺序后胜者一致才进入胜率；冲突为 unstable，无法判断为 abstained，失败为 missing。胜率按歌平均，平局计半胜；同时报告全部状态和有效分母。规则达标不直接决定文本偏好，纯文本评审不产生真实旋律贴合分。

失败调用和格式非法的评审会留档；使用相同命令配合 `--resume` 只补失败任务，不重新调用已成功的任务。相同设置包含模型、提示词、种子、段落上限和输入。旧三组评测目录不能作为本次四组评测目录续跑。更换评审模型请使用新的 `--judge-name`；改变已有记录的设置会被拒绝。代码变化后需新建评测目录。

运行期间会定期刷新 `judges/<评审名>/summary.json` 的进度，单个任务的响应和元数据随完成立即保存。续跑时可降低并发数；这不会重算成功任务，也不会改变已缓存的评分。

## 3. 本地 COMET-QE

可选依赖与模型运行环境由使用者显式安装：

```bash
python -m pip install -e ".[evaluation]"
python -m evaluate --resume <评测目录> --comet
```

默认使用论文同款 `Unbabel/wmt22-cometkiwi-da`，CPU、batch size=8。首次运行可能需要接受 Hugging Face 模型访问条款、登录及下载权重/编码器配置；程序不自动接受条款。模型权重、hparams 哈希、实际解析的模型 revision 和依赖版本记录在 `comet.json`。

```bash
# 固定 Hugging Face 的模型版本
python -m evaluate --resume <评测目录> --comet --comet-revision <commit-sha>

# 使用已准备的本地权重和编码器缓存；不联网下载
python -m evaluate --resume <评测目录> --comet \
  --comet-checkpoint /path/to/model/checkpoints/model.ckpt --comet-local-only
```

本地 checkpoint 的上两级目录需有 `hparams.yaml`。可用 `--comet-gpus` 和 `--comet-batch-size` 调整推理配置。仅启用离线规则或 LLM 评审时，不导入 COMET。

COMET-QE 对每个有可靠译词且需翻译的乐句输入 `src`、`mt`，不使用参考译文或预算。报告全量可用结果及共同歌曲上的宏平均。该指标是通用翻译质量诊断，不是实际可唱性评分。

## 后续扩展：实际可唱性记录（不参与当前四项评价）

纯文本模式保持 `not_evaluated`。实际试唱后，可按 `human-template.json` 填写：

```json
{
  "protocol": "同一伴奏与演唱条件；旋律贴合1差5好，跟唱难度1容易5困难",
  "ratings": [
    {
      "song": "abuku",
      "section": "V1",
      "variant": "workflow-rag",
      "rater": "rater-01",
      "melody_fit": 4,
      "singing_difficulty": 2
    }
  ]
}
```

```bash
python -m evaluate --resume <评测目录> --human-ratings /path/to/ratings.json
```

程序校验记录指向的候选、评分范围和重复评审，分别汇总 melody_fit（越高越好）与 singing_difficulty（越低越好），不会与 LLM 评分混合。稀疏、不配对的真人样本仅作描述性结果；记录身份与真实性由研究者核实。

完整指标依据及限制见 [评测方案](../docs/evaluation-plan.md)。测试采用 mock / 假推理后端，不调用付费接口或下载模型。
