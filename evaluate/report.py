"""报告明确区分规则分、模型评审和真人演唱记录。"""

from .common import DIMENSIONS, VARIANT_LABELS, read, write, variants_of

DIMENSION_LABELS = {"semantic": "核心语义", "emotion": "情绪意境", "naturalness": "自然度", "expressiveness": "表现力"}


def _number(value, percent=False):
    if value is None:
        return "—"
    return f"{value:.2%}" if percent else f"{value:.3f}"


def publish(directory, snapshot, rules):
    variants = variants_of(snapshot)
    group_name = {4: "四", 5: "五", 6: "六"}.get(len(variants), str(len(variants))) + "组"
    judges = {p.parent.name: read(p) for p in sorted((directory / "judges").glob("*/summary.json"))}
    comet = read(directory / "comet.json") if (directory / "comet.json").exists() else {"status": "not_evaluated"}
    human = read(directory / "human.json") if (directory / "human.json").exists() else {"status": "not_evaluated", "reason": "纯文本模式不评价实际可唱性"}
    meta = read(directory / "meta.json")
    rag_rule_differences = []
    for item in rules["paired_differences"]:
        if item["a"] == "workflow" and item["b"] == "workflow-rag":
            ci = item.get("ci95")
            difference = item.get("difference_a_minus_b")
            rag_rule_differences.append({**item, "a": "workflow-rag", "b": "workflow",
                                         "difference_a_minus_b": -difference if difference is not None else None,
                                         "ci95": [-ci[1], -ci[0]] if ci else None})
    summary = {"schema_version": 2, "variants": list(variants),
               "metric_categories": ["syllable_constraints", "text_quality", "relative_quality", "comet_qe"],
               "input_hash": snapshot["input_hash"], "status": meta["status"],
               "all_songs": rules["all_songs"], "paired_songs": rules["paired_songs"], "paired": rules["paired"],
               "paired_differences": rules["paired_differences"], "judges": judges, "comet_qe": comet,
               "rag_vs_workflow_rule_differences": rag_rule_differences,
               "actual_singability": human, "error": meta.get("error")}
    write(directory / "summary.json", summary)
    write(directory / "per-song.json", [{k: v for k, v in row.items() if k != "phrases"} for row in rules["per_song"]])
    lines = [f"# {group_name}译配结果评测", "", f"- 状态：`{meta['status']}`；输入快照：`{snapshot['input_hash']}`。",
             f"- 目标 {len(snapshot['songs'])} 首；{group_name}共同完成 {len(rules['paired_songs'])} 首。",
             "- 比较 " + "、".join(VARIANT_LABELS[v] for v in variants) + "。",
             "- 各组来源路径和输入版本核实范围见 inputs.json；保留分组名称，不将两条 workflow 合并为 ours。",
             "- 预算及硬规则使用相同计数器复算；宏平均按歌曲等权。缺失不填成 0 分，覆盖率另报。",
             "- 主流程历史未显式记录的生成配置不能推定；详见 inputs.json 的来源元数据。",
             "- 原样组的源文计数目标与人工预算不同，以下预算分衡量本项目目标，不是原论文节奏距离。", "",
             "## 音节约束", "",
             f"预算、距离及偏短偏长采用{group_name}共同完成歌曲。全库完成率与覆盖率仅作缺失诊断。", "",
             "| 方法 | 完成歌曲 | 乐句覆盖 | 预算通过率 | 平均预算距离 | 偏短 | 偏长 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v in variants:
        all_rows, paired = rules["all_songs"][v], rules["paired"][v]
        lines.append(f"| {v} | {all_rows['completed_songs']}/{all_rows['songs']} | {all_rows['covered_phrases']}/{all_rows['expected_phrases']} | {_number(paired['budget_pass_rate'], True)} | {_number(paired['mean_budget_distance'])} | {_number(paired['under_budget_rate'], True)} | {_number(paired['over_budget_rate'], True)} |")
    lines.extend(["", "分组含义：" + "；".join(f"`{v}`：{VARIANT_LABELS[v]}" for v in variants) + "。",
                  "RAG 增益主要比较 workflow-rag 与 workflow；若生成模型、提示词、温度、思考开关或运行版本不同，差值不能直接归因为 RAG。"])
    lines.extend(["", "硬规则通过率及逐句诊断见 [rules.json](rules.json)。字数达标不代表预算标注准确或实际可唱。", "",
                  "## 文本质量：LLM 匿名评审", ""])
    if not judges:
        lines.append("未评测。使用 --judge-dry-run 预览请求，--judge --client api 才调用模型。没有生成或补写任何模型评分。")
    for name, result in judges.items():
        selection = result["selection"]
        lines.extend([f"### {name}", "", f"状态 `{result['status']}`；选取 {selection['selected_sections']}/{selection['planned_sections']} 段；成功 {result['completed_tasks']}/{result['task_count']} 个任务。",
                      "", f"评分为模型判断，不是人评。下表各维度只比较{group_name}都有有效分数的相同段落，再按歌平均。", "",
                      "| 方法 | 核心语义 | 情绪意境 | 自然度 | 表现力 |", "| --- | ---: | ---: | ---: | ---: |"])
        for v in variants:
            cells = [f"{_number(result['absolute_paired'][v][d]['macro_score'])}（{result['absolute_paired'][v][d]['valid_sections']}/{result['absolute_paired'][v][d]['planned_sections']}段）" for d in DIMENSIONS]
            lines.append("| " + v + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## 相对质量：成对偏好", ""])
    if not judges:
        lines.append(f"未评测。{group_name}形成 {len(variants) * (len(variants) - 1) // 2} 对比较，每对交换候选顺序两次评审；不生成占位胜率。")
    for name, result in judges.items():
        lines.extend([f"### {name}", "", "| 方法 A / B | A 偏好胜率（歌宏平均） | 顺序稳定 | 不稳定 | 弃评 | 缺失 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for p in result["preferences"]:
            lines.append(f"| {p['a']} / {p['b']} | {_number(p['win_rate_a'], True)} | {p['stable_pairs']}/{p['planned_pairs']} | {p['unstable_pairs']} | {p['abstained_pairs']} | {p['missing_pairs']} |")
        focus = result.get("rag_vs_workflow")
        if focus:
            lines.append(f"\n重点比较：RAG 相对无 RAG 的偏好胜率 {_number(focus['win_rate_a'], True)}，顺序稳定 {focus['stable_pairs']}/{focus['planned_pairs']} 对；这一偏好仅针对有效文本比较。")
        lines.append("\n交换顺序复评后结论一致才进入胜率，平局计半胜；不稳定和缺失比例须同时阅读。匿名映射、短理由和乐句依据见 judges/ 中的逐任务记录。")
    lines.extend(["", "## 语义诊断：COMET-QE", ""])
    if comet["status"] == "skipped":
        lines.append("本次跳过 COMET-QE：" + comet.get("reason", "按用户要求跳过") + "。")
    elif comet["status"] != "completed":
        lines.append("未评测。默认不导入 COMET 或下载权重；显式 --comet 后在本地运行。")
    else:
        lines.extend([f"模型 `{comet['config']['model']}`；固定权重信息见 comet.json。共同歌曲的逐歌宏平均：", "",
                      "| 方法 | COMET-QE | 有效歌曲 / 乐句 |", "| --- | ---: | ---: |"])
        for v in variants:
            p = comet["paired"][v]
            lines.append(f"| {v} | {_number(p['macro_score'])} | {p['valid_songs']} / {p['phrases']} |")
        lines.append("\n该分数仅诊断通用翻译质量，不读取旋律和人工预算，艺术改写的合理性需单独判断。")
    if human["status"] == "completed":
        lines.extend(["", "## 附录：显式导入的真人记录（不计入本次四项指标）", ""])
        lines.extend(["下列为显式导入的真人记录，评分条件和覆盖率见 human.json；稀疏样本不视为配对结论。", "",
                      "| 方法 | 旋律贴合度 ↑ | 跟唱难度 ↓ |", "| --- | ---: | ---: |"])
        for v in variants:
            lines.append(f"| {v} | {_number(human['summary'][v]['melody_fit']['macro_score'])} | {_number(human['summary'][v]['singing_difficulty']['macro_score'])} |")
    lines.extend(["", "## 逐歌预算通过率", "", "| 歌曲 | " + " | ".join(variants) + " |", "| --- | " + " | ".join("---:" for _ in variants) + " |"])
    for item in snapshot["songs"]:
        cells = []
        for v in variants:
            r = next(r for r in rules["per_song"] if r["song"] == item["directory"] and r["variant"] == v)
            cells.append(_number(r["budget_pass_rate"], True) + ("（部分结果）" if not r["completed"] else ""))
        lines.append("| " + item["directory"] + " | " + " | ".join(cells) + " |")
    failures = [r for r in rules["per_song"] if not r["completed"]]
    if failures:
        lines.extend(["", "## 未完成与缺失", ""])
        for r in failures:
            reason = r.get("error") or "; ".join(r["issues"]) or r["source_status"]
            lines.append(f"- `{r['song']} / {r['variant']}`：覆盖 {r['covered_phrases']}/{r['expected_phrases']} 句；{reason}")
    if meta.get("error"):
        lines.extend(["", "## 阶段错误", "", str(meta["error"])])
    lines.extend(["", "歌曲配对差值与 95% bootstrap 区间见 [summary.json](summary.json)，其区间不涵盖全部模型采样波动。",
                  "各指标独立报告，不合成总分。部分评审或 COMET 阶段未运行，不影响已完成的离线结果。", "",
                  "[输入快照](inputs.json) · [复现清单](manifest.json) · [逐歌指标](per-song.json)"])
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
