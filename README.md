## 项目名称：
- 将日语歌的歌词译配为汉语，保证意境与情感一致，以及可唱性
## 项目解释：
- 所谓“译配”，就是既要翻译原歌词的意思，又要像填词一样调整字句，让中文音节与旋律、节拍严丝合缝，保证能跟着原曲唱出来。
	- 区别于普通翻译：普通翻译只求语义准确，不管字数是否合拍；而译配必须让译词“配”上原曲，具备可唱性，音节、重音、押韵都要与旋律匹配。
	- 区别于随意填词：填词可以写全新内容，与原词无关；译配则要求语义基本忠实于原歌词，是在原意基础上的“配曲”创作。
- 日语译配的难点：在表达相同意义的前提下，日语句子音节数往往比汉语句子音节数多。因此日语歌词以大意翻译的形式翻译成汉语后，译词字数往往会比原词少。而根据翻译目的改变，在追求原词语用价值的还原的情况下，音节数的补完成为译配的重点。在填补音节数的差距时，无可避免地需要添加语词，甚至增加整个句子。如何使添加了字词的译词恰到好处地传达原词感情正是日语歌词汉译的难点。
- 但我们可以放宽要求：
	- 歌曲的翻译并不一定需要让译词在语义上完全“忠于”原歌词。歌曲表达的主要是意境、情感。如果意境和情感传达到位，即使译词使用了完全不同的意象，表达了完全不同的语义，也算是好的翻译。由于日语音节数更多，中文译词更多情况下需要补充词语，增加意象等。这些增加的意象，可以是原歌词中完全没有出现过的，但只要能表达相同的意境和情感，也是可以的。
	- 译词的句子不需要符合语法规则。AI 喜欢输出语法元素齐全的句子，但其实歌词中缺少主语，缺少动词，组不成通顺的句子都没关系。只是单纯的意象堆叠也没关系。
	- 该项目最注重两个点是意境和情绪符合原曲，以及可唱性。可唱性最重要的就是歌词能跟着节拍唱出来，诸如押韵、重音之类的可以不考虑，它们并不是可唱性的必要元素，而且会让丰富多样的意象因为这些规则而无法使用。
## 方法
- 人工构建译词音节数量的约束。“能够跟着节拍唱出来”，并不意味着译词和原词的音节数要完全相同，最重要的是旋律中出现的音符个数与节拍。举例：有可能两个音节在旋律中是连读的，只占用了一个音符的时长，那只需要使用一个音节来对应即可；又或者，如果旋律中的两个十六分音符对应了两个音节，在对听感影响不大的情况下，合并成一个八分音符，并使用一个音节唱出来也是可以的。但是 AI 无法理解音乐旋律，所以这部分需求需要人工构建。
- 可以把不同性质的工作交给不同的 agent，让每个 agent 专注于自己的工作。这个项目有两种任务，一种负责感受歌词的意境和情感，需要思维发散开，就像一个人在 emo 一样；而另一种用来完成翻译任务，确保翻译出来的歌词符合要求，是一种比较严谨的任务。让这两种任务分开是有好处的。
	- 体验情感的 agent：比如可以让一个 agent 来头脑风暴相关的意象情感；可以准备一个华语歌曲歌词的数据集，让一个 agent 学习类似的意境和情感在华语歌曲中一般是怎么表达的；甚至可以让 agent 根据歌词写一段故事，作为翻译的意境和情感参考。
	- 用于翻译的 agent：确保可唱性的因素，比如音节数量，押韵，词语有没有被旋律分割等等；
- 使用 workflow。可以遵循 “体验情感，寻找意向 -> 完成翻译 -> 检查翻译是否满足可唱性约束” 这样的一个 workflow，如果翻译不满足需求，就在此基础上修改。让 AI 在多个轮次中不断完善翻译。
## 评测
- 评测无需严格的量化。本项目不注重严谨的科学性，而是更加注重实用性。歌词的意境，情感，甚至可唱性都是难以量化的。最终的评判，应该是真人来感受歌词并试着唱出来，然后给译词打分，因此评测比较主观。

## Workflow
-  使用三个 Agent A B C，分别用来处理情感意象，产生译词，以及检查译词是否满足约束。其中 A 和 B 使用 AI，C 基于规则。
- A 拿到歌词原文后，体会其中的意象与情感，用合适的方式表达出来作为输出传给 B；B 拿到歌词原文，每一句歌词应该产生的音节数（这一部分由人工标注），以及 A 的输出，然后基于此生成一份译文；C 拿到 B 的译文后，检测每一句歌词是否满足了可唱性约束。如果不满足的话，将不满足的地方传输 B，让 B 修改自己的输出。
- 可以使用 LangGraph 作为框架。
- 当前版本使用普通 Python 实现上述编排，尚未引入 LangGraph；A、B 调用模型，C 使用确定性的规则检查，每段默认最多生成 / 检查 3 轮。

## 实现
- 代码在 `dubbing/`，规范见 [docs/data-model.md](docs/data-model.md)，操作见 [docs/usage.md](docs/usage.md)。
- 模型调用已接通 DeepSeek 的 Anthropic 兼容端点（`dubbing/llm.py`）。默认使用 `pending` 客户端，在需要模型调用的位置保存请求并暂停；配置 API Key 后加 `--client api` 才真正调用，也可以通过 `DUBBING_CLIENT` 环境变量指定客户端。
- `final_20_song` 分支已收录 **20 首歌曲的 21 个数据版本，共 432 个乐句**。其中 `love-2000` 和 `love_2000` 是 `LOVE 2000` 的两套分句与预算版本，均予以保留。
- 本分支同时保存各首歌的译词、当前运行状态和历史运行记录。当前快照的 432 个乐句均为 `pass` 或 `skip`；这只表示通过程序规则，最终仍需真人试唱评判。
- 批量导入标注文本：`python scripts/import_lyrics.py data/lyrics.txt`，
  输入格式与选项见 [docs/usage.md](docs/usage.md)。

## 快速开始

需要 Python ≥ 3.11，运行依赖为 PyYAML、Requests。建议从源码以可编辑方式安装，并在项目根目录执行命令，以便读取歌曲和提示词文件。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# 不需要 API Key，即可查看本分支已有的歌库与结果
python -m dubbing songs
python -m dubbing view --song kotozute
```

首次配置时，将 [.env.example](.env.example) 复制为 `.env`，填写 `DEEPSEEK_API_KEY`。命令行启动时会自动读取 `.env`，且不覆盖终端中已有的同名环境变量。真实密钥不入库。

```bash
cp .env.example .env
# 编辑 .env，填写自己的 API Key
```

## 常用命令

```bash
# 单曲重新生成，归档保留旧运行记录
python -m dubbing run --song kotozute --client api --keep-history

# 全库重新生成；
python -m dubbing run --all --client api --keep-history

# 仅处理尚未通过的段落，复用已有 A 分析
python -m dubbing run --all --client api --skip-done

# 只运行指定的几首
python -m dubbing run --song kazeyo --song spiral --client api --keep-history

# 仅重跑一个段落；加 --continue 可使用该段上一版译词继续修改
python -m dubbing run --song shiori --only C1 --client api

# 预览当前状态下的初始请求，不调用模型
python -m dubbing run --all --dry-run

# 查看分析、历史记录和当前译词
python -m dubbing show --song kotozute --agent a
python -m dubbing runs --song kotozute
python -m dubbing view --song kotozute
```

真实运行默认会删除目标歌曲的旧 `runs/` 与 `state.yaml` 后从头生成；`--keep-history` 会把旧 run 移入 `archive/`，但仍会重建当前状态。`--only`、`--continue`、`--skip-done` 都保留已有记录。各选项的详细语义见 [docs/usage.md](docs/usage.md)。

`--dry-run` 会写入 `preview/`，可能复用已有 A 分析，不预测后续返修次数，因此不能作为完整重跑的精确请求数或费用估计。

## 数据准备与运行结果

- 新歌放在 `data/songs/<song_id>/song.yaml`，按 `song / sections / phrases` 组织。人工确认乐句顺序和音节预算，重复演唱的歌词也分别列出。
- `budget` 支持固定值和区间。可选的 `pattern` 各项须为正整数，合计必须落在预算范围内，否则加载会报错；它目前只为 B 提供参考，C 不检查译词是否按该分组断句。
- `keep_source: true` 仅支持整句保留原文，此时预算必须为空，也不能填写 `pattern`。目前尚不支持行内片段保留，括号内的英文不会自动从计数中排除。
- `scripts/import_lyrics.py` 将已标注的「歌词 总字数 [分组字数...]」文本转换为 YAML，不会根据原始歌词自动确认预算。可先运行 `python scripts/import_lyrics.py data/lyrics.txt --dry-run --strict` 检查输入；已有文件默认跳过，`--force` 才会覆盖。
- 当前程序不读取音频、MIDI 或乐谱来自动校准预算。文本估算需要人工对照演唱复核，不能将日语音拍数或音符数直接当作已经确认的中文可唱字位。
- 每首歌完成运行后，自动写出 `lyrics.zh.txt`（演唱文本，含保留的原文）与 `lyrics.md`（原文、译词、字数、预算和状态对照）。
- `state.yaml` 保存当前 A / B / C 产物的指针；`runs/` 保存请求、解析后的输出、元数据，以及已取得的 A / B 模型正文 `response.txt`。失败记录可能没有正文或输出文件。
- 本分支已将运行结果作为快照纳入 Git；`.gitignore` 仍默认忽略新增运行文件，后续若要提交这些文件，需显式选择并使用 `git add -f`。手动编辑导出歌词不会更新 run 或状态，再次导出会按状态重新生成。

## 开发与测试

### 论文双 baseline

另提供论文 Exp.4 的两组对照：`paper-original` 保留原提示、不传人工标注；`paper-annotated` 加入人工预算区间和分组。两组共用当前模型配置，按段生成并独立保存结果，不覆盖已有单提示词 baseline 或主流水线产物。

```bash
python baseline/run_paper_baselines.py --all --variant both --dry-run
python baseline/run_paper_baselines.py --all --variant both --client api
```

“原样”指提示和输入方式，当前模型与清理步骤的差异会明确记录，不代表论文整套实验的严格复现。用法、输出和出处见 [baseline/PAPER_BASELINES.md](baseline/PAPER_BASELINES.md)。

### 测试

```bash
python -m pytest
```

测试使用临时数据、脚本化响应或 mock，不调用真实模型 API。提示词集中在 `prompts/shared.md`、`prompts/a.md`、`prompts/b.md`；歌曲数据与提示词分开维护。数据设计见 [docs/data-model.md](docs/data-model.md)，完整操作说明见 [docs/usage.md](docs/usage.md)。
