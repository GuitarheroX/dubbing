```
阅读 README.md 理解这个项目的任务。首先，A 和 B 会有一个总的提示词来描述它们的任务；其次，A 输入的应该是完成的歌词，B 的输入应该是完整歌词的每一句，以及这一句应该包含多少音节，以及 A 的输出。当前的问题是数据应该以什么形式组织？无需实现，和我商讨一个方案
1. “句”就是乐句，每个乐句都会标注一个音节数，不需要 pattern 再来分割乐句。 2. B 应该是整段生成的。B 在翻译当前句的时候， 需要参考这个乐句所在的整个 section，以及它已经翻译出来的内容。 3. 不用 4. 用区间吧，但是MVP 先让区间退化为定值 5. 不需要行内词对齐。我们对单独一句歌词的语义是否保留下来并不关注。 另外还有几点 1. A 的输出先不考虑 references，然后不要包含 do_not。do_not 里的提示词应该在 B 的总任务提示词里。然后“imagery_pool”里的 "写法建议" 也是。然后 line_hints 里不要包含 avoid。总之就是数据里不要限制 B 的生成。 2. song.yaml 里先不考虑 borrow 3. B 的输出里不要有 unused_imagery，是否使用前文已经使用过的意象交给 agent 自己判断。
```

```
首先我改了一些提示词，已经提交了。 我先在想重新考虑 pattern 约束，也就是在 song.yaml 里，指定一句歌词的可以分成几个部分，分别都有多少音节，比如 pattern 为 3, 4, 5 就是这一句词一共 12 个音节，分成三部分，每部分分别是三个四个和五个音节。然后默认应该是一句歌词只有一部分。但是先不要约束 agent b 的行为，只需要让他看到 pattern 就行。
```

```
我想在这个项目的 workflow 方案的基础上实现 RAG 功能，命名为 workflow-RAG 方案，用于与当前的 workflow 做对比。
我想让 AI 进行歌词生成时，能够以真人写出来的现代诗作为参考，因为他们往往有更合理或更有想象力的意象情感以及表达方式，以及更通顺的表达方式。我希望 AI 写作时能够模仿现代诗的句子，所以我需要你帮我构建一个方案：
1. 设计提示词与工作流程，让 agent B 在生成一段歌词时，能够把自己的想法用于检索现代诗数据库，从中选取诗句并作为写作参考。
2. 设计 RAG 构建全流程。需要考虑数据集的预处理，文本切块，使用服务器跑文本嵌入，并把嵌入后的向量存入向量库。
	1. 文本切块应该抛弃固定 token 数，以“诗节（stanza）”为最小完整单元。一个诗节为一个 chunk，保留诗题作为 chunk 的 metadata，不要设置重叠。如果某个诗节特别长，再通过 token 数拆分。一个 chunk 的 token 数上限设置为 256。
	2. 我有一个服务器可以用 GPU 跑词嵌入，实现代码时可以考虑 GPU。
向量数据库：Chrome
现代诗数据集：sheepzh/poetry
文本嵌入模型：bge-large-zh-v1.5
先不用写代码实现，和我讨论一下方案。
```


```
统一一下各个 baseline 与我们自己写的 workflow 的输出组织方式： 0. 首先清理一下 workflow 方案产生的 runs 记录，直接删掉。 1. baseline B 中的 sections 里的运行信息，像 workflow 一样只保留 meta.json，output.json 和 request.json 就够了。目录应该是 sections/001/cleanup/meta.json，output.json 和 request.json，然后 002 再放这些东西。然后再把001 和 002这些段落目录下的 alignment.json，output.json 和 section.json 去掉。最后再把 run_id 目录里的 config.json，meta.json。output.json，report.json 和 song.json 去掉。 2. workflow 的 runs 目前是每一段都产生一个运行记录。改成 baseline B 的形式，每次跑这一首歌产生一次运行记录。然后也按照 sections/001/A/三个文件的格式储存运行记录。同时，workflow 现在是独占一个歌名目录的，但是今后可能还会有 workflow 的改进版，所以为了方便管理，把它放在叫做 workflow 的子目录里吧。 3. baseline A 目前只生成一个 txt 文件。让它像 baseline B 和 workflow 一样保留运行信息。其次输出格式也和 workflow 统一一下，删除它当前的输输出格式限制，让它的输出和 workflow 一样输出 lyrics.zh.txt 和 lyrics.md。然后把这些东西放在 baselines/baseline_A 吧。 4. 统一一下 baseline 调用的入口吧，应该有一个脚本可以指定运行 baseline A 或者 baseline B，而不是 baselineA 和 baselineB 各自有一个脚本 ps：改完后，每个歌名目录下应该有两个子目录，一个 baselines，一个 workflow。所以今后如果有 workflow_1，也会和他们并列放在歌名目录里 ps：如果要调用 api 测试的话，测一首歌的就够了。
```

```
你构造的检索逻辑太复杂了，不要用那个。按照我下面说的做吧。
- 首先改一下三个 agent 的名字。A 的名字改成 perceiver，B 改成 writer，C 改成 reviewer。
- 然后新加入一个 agent 叫做 mimic，它负责检索诗句并参考或模仿。
	- mimic 内部分为两个部分，第一部分是检索，直接用 writer 写好的歌词段落来检索 RAG；第二部分是写作，也就是参考检索出来的诗句，修改 writer 写出的歌词，产生第二版歌词。
	- mimic 负责严格对齐音节数量，因此 writer 不再需要严格的提示词来要求它对齐音节数量，只需要在提示词中简单提一下就行，类似“译文大概在 xx 字左右”。
- reviewer 现在应该是拿到 mimic 写出的结果。它现在应该把评审结果交给 mimic 的写作部分。也就是说如果结果有问题，那么只需要让 mimic 重新写作就行，不需要让它重新检索。
- 整体的 workflow 应该是这样的：perceiver 体会整首歌后把结果给 writer，writer 写出初版歌词交给 mimic，mimic 根据初版歌词检索诗句，并参考诗句写出第二版歌词交给 reviewer。reviewer 判断是否合格，如果合格就产出最终版歌词，如果不合格，就再交给 mimic 的写作部分，让它根据评审意见重新写作。
sheepzh/poetry 项目我已经下载到 ~/projects/poetry 了
```
