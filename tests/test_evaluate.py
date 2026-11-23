import json
from pathlib import Path

import pytest
import yaml

from dubbing.llm import ScriptedClient
from dubbing.models import Song
from dubbing.recorded_client import RecordedAPIClient
from dubbing.store import Request, RunStore
from evaluate.cli import main
from evaluate.common import DIMENSIONS, VARIANTS, digest, read, write
from evaluate.comet_qe import run_comet
from evaluate.human import load_ratings
from evaluate.inputs import aligned, collect
from evaluate.judges import build_tasks, execute_task, run_judges, summarize, validate_result
from evaluate.rules import evaluate_rules


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "songs"
    directory = root / "fixture-dir"
    raw = {
        "song": {"id": "internal-id", "title": "测试", "artist": "", "source_lang": "ja"},
        "sections": [{"id": "V1", "role": "verse"}, {"id": "C1", "role": "chorus"}],
        "phrases": [
            {"id": "V1-01", "section": "V1", "ja": "あ", "budget": {"min": 2, "max": 3}, "pattern": [1, 1]},
            {"id": "V1-02", "section": "V1", "ja": "い", "budget": None},
            {"id": "V1-03", "section": "V1", "ja": "La", "budget": None, "keep_source": True},
            {"id": "C1-01", "section": "C1", "ja": "う", "budget": {"min": 2, "max": 2}},
        ],
    }
    directory.mkdir(parents=True)
    (directory / "song.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True))
    song = Song.from_json(raw)
    state = {"sections": {}}
    for sec, texts in (("V1", {"V1-01": "风 雨", "V1-02": "夜"}), ("C1", {"C1-01": "月光"})):
        run = directory / "runs" / f"b-{sec}"
        write(run / "output.json", {"section": sec, "phrases": [{"id": k, "zh": v, "chars": 999} for k, v in texts.items()]})
        write(run / "request.json", {"data": {"focus": sec, "phrases": [p.to_json() for p in song.phrases]}})
        write(run / "meta.json", {"model": "fixture-generator", "temperature": .7, "round": 1})
        state["sections"][sec] = {"draft": run.name, "round": 1}
    (directory / "state.yaml").write_text(yaml.safe_dump(state))
    rag_store = RunStore(directory / "workflow_rag")
    rag_run = rag_store.new_run()
    for index, sec in enumerate(song.sections, 1):
        rag_store.prepare(rag_run, "mimic_write", Request("fixture", "fixture", {"phrases": [p.to_json() for p in song.phrases]}), section_index_=index)
        rag_store.finish(rag_run, "mimic_write", {"section": sec.id, "phrases": [{"id": p.id, "zh": "云" if p.id == "V1-02" else "星 月"} for p in song.translatable_of(sec.id)]}, meta={"model": "fixture-generator"}, section_index_=index)
    rag_store.set_latest(rag_run, pipeline="rag", status="completed")
    batch = []
    for variant in VARIANTS[:2]:
        run = directory / "baselines" / variant / "run-1"
        write(run / "song.json", song.to_json())
        write(run / "config.json", {"variant": variant, "model": "fixture-generator"})
        write(run / "meta.json", {"status": "completed"})
        all_rows = []
        for index, sec in enumerate(song.sections, 1):
            rows = [{"id": p.id, "zh": "保留" if p.keep_source else "风" if variant == "paper-original" and p.id == "V1-01" else "风雨雷电" if p.id == "V1-01" else "夜色"} for p in song.phrases_of(sec.id)]
            write(run / "sections" / f"{index:03d}" / "output.json", {"section": sec.id, "phrases": rows})
            all_rows.extend({"id": r["id"], "model_text": r["zh"]} for r in rows)
        write(run / "output.json", {"phrases": all_rows})
        write(run.parent / "latest.json", {"run": run.name})
        batch.append({"song": directory.name, "variant": variant, "directory": str(run)})
    batch_path = tmp_path / "batch.json"
    write(batch_path, {"results": batch})
    return root, directory, batch_path


def test_collect_and_rule_denominators(dataset):
    root, directory, batch = dataset
    snapshot = collect(root, batch=batch)
    assert snapshot["songs"][0]["song"]["song"]["id"] == "internal-id"
    assert snapshot["songs"][0]["directory"] == "fixture-dir"
    result = evaluate_rules(snapshot, samples=20)
    assert result["paired_songs"] == ["fixture-dir"]
    a, b, workflow, ours = [result["paired"][v] for v in VARIANTS]
    assert a["evaluated_phrases"] == 2 and a["budget_pass_rate"] == .5
    assert a["mean_budget_distance"] == .5 and a["under_budget_rate"] == .5
    assert b["over_budget_rate"] == .5
    assert ours["budget_pass_rate"] == 1 and ours["rule_pass_rate"] == 1
    assert ours["expected_phrases"] == 3  # 无预算句进入覆盖率，保留句不进入。
    assert ours["phrase_coverage"] == 1


def test_partial_baseline_uses_only_aligned_sections(dataset):
    root, directory, _ = dataset
    run = directory / "baselines/paper-original/run-1"
    (run / "sections/002/output.json").unlink()
    (run / "output.json").unlink()
    write(run / "meta.json", {"status": "failed", "error": "行数错误"})
    (run / "lyrics.zh.txt").write_text("〈未译〉不能算译词")
    snapshot = collect(root)
    rules = evaluate_rules(snapshot, samples=20)
    original = rules["all_songs"]["paper-original"]
    assert original["covered_phrases"] == 2 and original["phrase_coverage"] == 2 / 3
    assert original["song_completion_rate"] == 0 and rules["paired_songs"] == []
    row = next(r for r in rules["per_song"] if r["variant"] == "paper-original")
    assert next(r for r in row["phrases"] if r["id"] == "C1-01")["budget_distance"] is None


def test_duplicate_unknown_and_placeholder_are_not_coverage():
    issues = []
    texts = aligned([{"id": "x", "zh": "风"}, {"id": "x", "zh": "雨"},
                     {"id": "unknown", "zh": "月"}, {"id": "y", "zh": "〈未译〉"}], {"x", "y"}, issues, "V1")
    assert texts == {} and len(issues) >= 4


def test_rejects_changed_source_and_tampered_complete_output(dataset):
    root, directory, _ = dataset
    run = directory / "baselines/paper-original/run-1"
    full = read(run / "output.json")
    full["phrases"][0]["model_text"] = "被改过"
    write(run / "output.json", full)
    with pytest.raises(ValueError, match="不一致"):
        collect(root)
    frozen = read(run / "song.json")
    frozen["phrases"][0]["ja"] = "別"
    write(run / "song.json", frozen)
    with pytest.raises(ValueError, match="输入版本不同"):
        collect(root)


def test_stale_main_input_rejected(dataset):
    root, directory, _ = dataset
    p = directory / "runs/b-V1/request.json"
    value = read(p)
    value["data"]["phrases"][0]["pattern"] = [2]
    write(p, value)
    with pytest.raises(ValueError, match="输入版本不同"):
        collect(root)


def score_result(task, *, null_semantic=False):
    return {"candidates": [{"label": label, "scores": {
        d: {"score": None if null_semantic and d == "semantic" and variant == "workflow-rag" else 5 if variant == "workflow-rag" else 3,
            "reason": "对应乐句表达明确", "phrase_ids": [task["phrase_ids"][0]]}
        for d in DIMENSIONS}} for label, variant in task["mapping"].items()]}


def result_for(task):
    if task["kind"] == "score":
        return score_result(task)
    pair = set(task["mapping"].values())
    if pair == {"paper-original", "workflow-rag"}:
        winner = "A"  # deliberately position-dependent: should be unstable
    elif pair == {"paper-annotated", "workflow-rag"}:
        winner = "tie"
    else:
        preferred = max(pair, key=VARIANTS.index)
        winner = next(k for k, v in task["mapping"].items() if v == preferred)
    return {"winner": winner, "reason": "对应乐句更自然", "phrase_ids": [task["phrase_ids"][0]]}


def test_anonymity_and_order_mapping(dataset):
    root, _, _ = dataset
    snapshot = collect(root)
    tasks, selection = build_tasks(snapshot, ["fixture-dir"])
    assert selection == {"planned_sections": 2, "selected_sections": 2, "planned_tasks": 26}
    for task in tasks:
        assert all(v not in task["messages"]["user"] for v in VARIANTS)
        assert "analysis" not in task["messages"]["user"]
    outcomes = [{"status": "completed", "result": result_for(t)} for t in tasks]
    # 一个候选的某个维度弃评，配对绝对评分必须排除该段的四组该维度。
    outcomes[0]["result"] = score_result(tasks[0], null_semantic=True)
    result = summarize(tasks, outcomes)
    assert all(result["absolute_paired"][v]["semantic"]["valid_sections"] == 1 for v in VARIANTS)
    assert result["rag_vs_workflow"]["a"] == "workflow-rag"
    assert result["rag_vs_workflow"]["win_rate_a"] == 1
    for row in result["preferences"]:
        if row["b"] == "workflow-rag" and row["a"] == "paper-original":
            assert row["unstable_pairs"] == 2 and row["win_rate_a"] is None
        elif row["b"] == "workflow-rag" and row["a"] == "paper-annotated":
            assert row["win_rate_a"] == .5
        else:
            assert row["win_rate_a"] == 0


@pytest.mark.parametrize("score", [True, 0, 6, 2.5, "3"])
def test_score_validation_rejects_invalid_values(dataset, score):
    snapshot = collect(dataset[0])
    task = build_tasks(snapshot, ["fixture-dir"], limit=1)[0][0]
    value = score_result(task)
    value["candidates"][0]["scores"]["semantic"]["score"] = score
    with pytest.raises(ValueError, match="评分只能"):
        validate_result(value, task)


def test_evidence_and_labels_are_validated(dataset):
    task = build_tasks(collect(dataset[0]), ["fixture-dir"], limit=1)[0][0]
    value = score_result(task)
    value["candidates"][0]["scores"]["emotion"]["phrase_ids"] = ["fake"]
    with pytest.raises(ValueError, match="phrase ID"):
        validate_result(value, task)
    value = score_result(task)
    value["candidates"][0]["label"] = value["candidates"][1]["label"]
    with pytest.raises(ValueError, match="标签"):
        validate_result(value, task)


def test_judge_resume_retries_only_failed_task_and_archives(dataset, tmp_path):
    snapshot = collect(dataset[0]);rules = evaluate_rules(snapshot, samples=20)
    tasks = build_tasks(snapshot, rules["paired_songs"], limit=1)[0]
    responses = [json.dumps(result_for(t)) for t in tasks]
    first = ScriptedClient(["非法 JSON", *responses[1:]])
    out = tmp_path / "judge"
    result = run_judges(snapshot, rules, first, out, limit=1)
    assert result["failed_tasks"] == 1
    second = ScriptedClient([responses[0]])
    result = run_judges(snapshot, rules, second, out, limit=1)
    assert result["failed_tasks"] == 0 and not second._queue
    assert list((out / "tasks" / tasks[0]["id"] / "attempts").glob("*/meta.json"))
    third = ScriptedClient([])
    assert run_judges(snapshot, rules, third, out, limit=1)["completed_tasks"] == 13
    third.model = "different"
    with pytest.raises(ValueError, match="设置已变化"):
        run_judges(snapshot, rules, third, out, limit=1)


def test_transport_is_anonymous_and_credentials_never_saved(dataset, tmp_path, monkeypatch):
    task = build_tasks(collect(dataset[0]), ["fixture-dir"], limit=1)[0][0]
    key = "sk-" + "x" * 30
    seen = []
    class Response:
        status_code = 200
        def json(self):
            return {"content": [{"type": "text", "text": json.dumps(score_result(task))}], "stop_reason": "end_turn"}
    def post(url, json, headers, timeout):
        seen.append(json)
        assert headers["x-api-key"] == key
        return Response()
    monkeypatch.setattr("dubbing.llm.requests.post", post)
    client = RecordedAPIClient(api_key=key, thinking="disabled")
    result = execute_task(task, client, tmp_path / "task")
    assert result["status"] == "completed"
    assert seen[0]["thinking"] == {"type": "disabled"}
    assert all(v not in seen[0]["messages"][0]["content"][0]["text"] for v in VARIANTS)
    assert all(key not in p.read_text() for p in (tmp_path / "task").rglob('*') if p.is_file())


def test_comet_alignment_no_reference_kept_or_placeholder(dataset, tmp_path):
    snapshot = collect(dataset[0]);rules = evaluate_rules(snapshot, samples=20)
    calls = []
    def predictor(rows, **kwargs):
        calls.append(rows)
        assert len(rows) == 12
        assert all(r["id"] != "V1-03" and "ref" not in r for r in rows)
        return [.75] * len(rows), {"resolved_revision": "frozen-commit"}
    out = tmp_path / "comet.json"
    result = run_comet(snapshot, rules, out, predictor=predictor)
    assert result["paired"]["workflow-rag"]["macro_score"] == .75
    assert run_comet(snapshot, rules, out, predictor=predictor) == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="设置或输入已变化"):
        run_comet(snapshot, rules, out, revision="changed", predictor=predictor)


def test_comet_local_backend_uses_pinned_snapshot_and_records_versions(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from evaluate.comet_qe import local_predict
    model_dir = tmp_path / "frozen-revision"
    checkpoint = model_dir / "checkpoints/model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture-checkpoint")
    (model_dir / "hparams.yaml").write_text("fixture")
    calls = []
    def predict(rows, **kwargs):
        assert rows == [{"src": "あ", "mt": "风"}]
        assert kwargs == {"batch_size": 8, "gpus": 0}
        return SimpleNamespace(scores=[.8])
    model = SimpleNamespace(eval=lambda: None, predict=predict,
                            encoder=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash="encoder-revision"))))
    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(model_dir)
    def load(path, **kwargs):
        assert path == str(checkpoint) and kwargs == {"local_files_only": True}
        return model
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, "comet", SimpleNamespace(load_from_checkpoint=load))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr("evaluate.comet_qe.version", lambda name: "fixture-version")
    scores, metadata = local_predict([{"src": "あ", "mt": "风"}], model_name="fixture-model", revision="fixed-commit", checkpoint=None, local_only=True, batch_size=8, gpus=0, seed=0)
    assert scores == [.8] and calls[0]["revision"] == "fixed-commit"
    assert calls[0]["local_files_only"] is True
    assert metadata["resolved_revision"] == "frozen-revision" and metadata["encoder_revision"] == "encoder-revision"


@pytest.mark.parametrize("bad", [float('nan'), float('inf'), True])
def test_comet_rejects_nonfinite_scores(dataset, tmp_path, bad):
    snapshot = collect(dataset[0]);rules = evaluate_rules(snapshot, samples=20)
    with pytest.raises(ValueError, match="非有限"):
        run_comet(snapshot, rules, tmp_path / "comet.json", predictor=lambda rows, **kwargs: ([bad] * len(rows), {}))


def test_human_records_remain_separate(dataset, tmp_path):
    snapshot = collect(dataset[0]);p = tmp_path / "human.json"
    data = {"protocol": "同一伴奏，实际试唱；1容易5困难", "ratings": [{"song": "fixture-dir", "section": "V1", "variant": "workflow-rag", "rater": "r1", "melody_fit": 4, "singing_difficulty": 2}]}
    write(p, data)
    result = load_ratings(p, snapshot)
    assert result["summary"]["workflow-rag"]["melody_fit"]["macro_score"] == 4
    data["ratings"].append(data["ratings"][0]);write(p, data)
    with pytest.raises(ValueError, match="重复"):
        load_ratings(p, snapshot)


def test_cli_offline_snapshot_resume_and_no_overwrite(dataset, tmp_path, monkeypatch):
    root, directory, batch = dataset
    out = tmp_path / "evaluation"
    hashes = {str(p): p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    monkeypatch.setattr("evaluate.cli.RecordedAPIClient", lambda **kwargs: pytest.fail("离线模式不能创建 API 客户端"))
    assert main(["--root", str(root), "--batch", str(batch), "--out", str(out), "--bootstrap-samples", "20"]) == 0
    assert read(out / "human-template.json")["ratings"] == []
    assert read(out / "summary.json")["actual_singability"]["status"] == "not_evaluated"
    assert "LLM 匿名评审" in (out / "report.md").read_text()
    assert all(Path(p).read_bytes() == value for p, value in hashes.items())
    old = (out / "meta.json").read_bytes()
    assert main(["--root", str(root), "--out", str(out), "--bootstrap-samples", "20"]) == 2
    assert (out / "meta.json").read_bytes() == old
    assert main(["--resume", str(out), "--bootstrap-samples", "20"]) == 0
    snapshot = read(out / "inputs.json");snapshot["songs"][0]["directory"] = "tampered";write(out / "inputs.json", snapshot)
    old = (out / "meta.json").read_bytes()
    assert main(["--resume", str(out), "--bootstrap-samples", "20"]) == 2
    assert (out / "meta.json").read_bytes() == old


def test_cli_requires_explicit_api_and_preview_does_not_call(dataset, tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        main(["--judge"])
    monkeypatch.setattr(RecordedAPIClient, "complete", lambda *args, **kwargs: pytest.fail("预览不能调用 API"))
    out = tmp_path / "preview"
    assert main(["--root", str(dataset[0]), "--out", str(out), "--judge-dry-run", "--judge-limit", "1", "--bootstrap-samples", "20"]) == 0
    result = read(out / "judges/judge-1/summary.json")
    assert result["preview_tasks"] == 13 and result["completed_tasks"] == 0
    assert result["absolute"]["workflow-rag"]["semantic"]["macro_score"] is None


def test_cli_failed_judges_remain_partial_until_resumed(dataset, tmp_path, monkeypatch):
    root = dataset[0];out = tmp_path / "evaluation"
    tasks = build_tasks(collect(root), ["fixture-dir"], limit=1)[0]
    responses = [json.dumps(result_for(t)) for t in tasks]
    clients = []
    def make_client(**kwargs):
        client = ScriptedClient(["invalid", *responses[1:]] if not clients else [responses[0]])
        client.api_key = "fixture-key"
        clients.append(client)
        return client
    monkeypatch.setattr("evaluate.cli.RecordedAPIClient", make_client)
    assert main(["--root", str(root), "--out", str(out), "--judge", "--client", "api", "--judge-limit", "1", "--bootstrap-samples", "20"]) == 1
    assert read(out / "meta.json")["status"] == "partial"
    assert main(["--resume", str(out), "--bootstrap-samples", "20"]) == 1
    assert main(["--resume", str(out), "--judge", "--client", "api", "--judge-limit", "1", "--bootstrap-samples", "20"]) == 0
    assert read(out / "judges/judge-1/summary.json")["completed_tasks"] == 13


def test_cli_optional_stage_failure_preserves_offline_report(dataset, tmp_path, monkeypatch):
    out = tmp_path / "evaluation"
    def fail(*args, **kwargs):
        raise ValueError("缺少可选模型环境")
    monkeypatch.setattr("evaluate.cli.run_comet", fail)
    assert main(["--root", str(dataset[0]), "--out", str(out), "--comet", "--bootstrap-samples", "20"]) == 2
    assert read(out / "meta.json")["status"] == "failed"
    assert read(out / "rules.json")["paired"]["workflow-rag"]["budget_pass_rate"] == 1
    assert "缺少可选模型环境" in (out / "report.md").read_text()


def test_full_workflow_on_synthetic_data_without_network(dataset, tmp_path, monkeypatch):
    root = dataset[0];out = tmp_path / "complete-workflow"
    tasks = build_tasks(collect(root), ["fixture-dir"], limit=1)[0]
    client = ScriptedClient([json.dumps(result_for(t)) for t in tasks])
    client.api_key = "fixture-key"
    monkeypatch.setattr("evaluate.cli.RecordedAPIClient", lambda **kwargs: client)
    monkeypatch.setattr("dubbing.llm.requests.post", lambda *args, **kwargs: pytest.fail("流程测试不得联网"))
    monkeypatch.setattr("evaluate.cli.run_comet", lambda *args, **kwargs: run_comet(*args, **kwargs, predictor=lambda rows, **settings: ([.88] * len(rows), {"resolved_revision": "fixture-commit"})))
    ratings = tmp_path / "ratings.json"
    write(ratings, {"protocol": "虚构测试记录，不是真人评测", "ratings": [
        {"song": "fixture-dir", "section": "V1", "variant": "workflow-rag", "rater": "test-rater", "melody_fit": 4, "singing_difficulty": 2}
    ]})
    assert main(["--root", str(root), "--out", str(out), "--judge", "--client", "api", "--judge-limit", "1", "--comet", "--human-ratings", str(ratings), "--bootstrap-samples", "20"]) == 0
    summary = read(out / "summary.json")
    assert summary["status"] == "completed"
    assert summary["paired"]["workflow-rag"]["budget_pass_rate"] == 1
    assert summary["judges"]["judge-1"]["absolute_paired"]["workflow-rag"]["semantic"]["macro_score"] == 5
    assert summary["comet_qe"]["paired"]["workflow-rag"]["macro_score"] == pytest.approx(.88)
    assert summary["actual_singability"]["summary"]["workflow-rag"]["singing_difficulty"]["macro_score"] == 2
    assert not client._queue


@pytest.mark.parametrize("pipeline,old_names", [("workflow", False), ("workflow", True), ("rag", False)])
def test_reads_unified_workflow_and_rag_final_agent(dataset, pipeline, old_names):
    root, directory, _ = dataset
    song = Song.from_json(yaml.safe_load((directory / "song.yaml").read_text()))
    base = directory / ("workflow_rag" if pipeline == "rag" else "workflow")
    store = RunStore(base);run_id = store.new_run()
    agent = "mimic_write" if pipeline == "rag" else "writer"
    record_agent = "B" if old_names else agent
    for index, section in enumerate(song.sections, 1):
        request = Request("fixture", "fixture", {"phrases": [p.to_json() for p in song.phrases]})
        store.prepare(run_id, record_agent, request, section_index_=index)
        output = {"section": section.id, "phrases": [{"id": p.id, "zh": "山 风"} for p in song.translatable_of(section.id)]}
        store.finish(run_id, record_agent, output, meta={"model": "fixture-model"}, section_index_=index)
        if pipeline == "rag":
            store.finish(run_id, "writer", {"section": section.id, "phrases": [{"id": p.id, "zh": "初稿不是最终文本"} for p in song.translatable_of(section.id)]}, section_index_=index)
    # incomplete 可能只是 reviewer 未通过，生成完成与预算是否通过分开。
    store.set_latest(run_id, pipeline=pipeline, status="incomplete")
    snapshot = collect(root)
    candidate = snapshot["songs"][0]["candidates"]["workflow-rag" if pipeline == "rag" else "workflow"]
    assert candidate["source"]["layout"] == "unified" and candidate["source"]["draft_agent"] == agent
    assert candidate["source"]["record_agents"]["V1"] == record_agent
    assert candidate["texts"]["V1-01"] == "山 风" and candidate["completed"] is True
    assert snapshot["selection"]["roots"]["workflow-rag" if pipeline == "rag" else "workflow"] == str(root.resolve())


def test_reads_consolidated_paper_baselines_and_validates_annotation_inputs(dataset):
    from baseline.run_baselines import run_paper
    root, directory, _ = dataset
    song = Song.from_json(yaml.safe_load((directory / "song.yaml").read_text()))
    outputs = ["原文生成", json.dumps({"lines": ["山风", "夜色", "La"]}), "原文生成", json.dumps({"lines": ["月光"]})]
    for variant in VARIANTS[:2]:
        run_paper(song, variant, ScriptedClient(outputs.copy()), out=directory / "baselines" / variant)
    snapshot = collect(root)
    for variant in VARIANTS[:2]:
        candidate = snapshot["songs"][0]["candidates"][variant]
        assert candidate["completed"] and candidate["source"]["layout"] == "unified"
        assert candidate["texts"]["V1-01"] == "山风"
    store = RunStore(directory / "baselines/paper-annotated")
    request_path = store.record_dir(store.latest_run(), "generation", 1) / "request.json"
    request = read(request_path);request["data"]["phrases"][0]["budget"] = {"min": 1, "max": 1};write(request_path, request)
    with pytest.raises(ValueError, match="输入版本不同"):
        collect(root)


def test_unaligned_unified_baseline_does_not_count_placeholder_output(dataset):
    from baseline.run_baselines import BaselineOutputError, run_paper
    root, directory, _ = dataset
    song = Song.from_json(yaml.safe_load((directory / "song.yaml").read_text()))
    base = directory / "baselines/paper-original"
    with pytest.raises(BaselineOutputError):
        run_paper(song, "paper-original", ScriptedClient(["生成了四行", json.dumps({"lines": ["甲", "乙", "丙", "丁"]})]), out=base)
    store = RunStore(base);new_run = store.list_runs()[-1]
    store.set_latest(new_run, status="incomplete")
    candidate = collect(root)["songs"][0]["candidates"]["paper-original"]
    assert not candidate["completed"] and candidate["texts"] == {}
    assert any("尚未对齐" in message for message in candidate["issues"])


def test_four_groups_remain_distinct_and_have_six_reversed_pairs(dataset):
    snapshot = collect(dataset[0])
    assert snapshot["variants"] == ["paper-original", "paper-annotated", "workflow", "workflow-rag"]
    candidates = snapshot["songs"][0]["candidates"]
    assert candidates["workflow"]["texts"]["V1-01"] == "风 雨"
    assert candidates["workflow-rag"]["texts"]["V1-01"] == "星 月"
    tasks, _ = build_tasks(snapshot, ["fixture-dir"], limit=1)
    assert len(tasks) == 13
    score = next(t for t in tasks if t["kind"] == "score")
    assert set(score["mapping"]) == set("ABCD") and set(score["mapping"].values()) == set(VARIANTS)
    groups = {}
    for task in tasks:
        if task["kind"] == "pair":
            groups.setdefault(task["comparison"], []).append(task)
    assert len(groups) == 6
    for pair in groups.values():
        assert len(pair) == 2 and {p["orientation"] for p in pair} == {0, 1}
        assert pair[0]["mapping"]["A"] == pair[1]["mapping"]["B"]
        assert pair[0]["mapping"]["B"] == pair[1]["mapping"]["A"]


def test_missing_rag_excluded_from_joint_comparison_not_treated_as_workflow(dataset):
    root, directory, _ = dataset
    store = RunStore(directory / "workflow_rag")
    (store.record_dir(store.latest_run(), "mimic_write", 2) / "output.json").unlink()
    snapshot = collect(root);rules = evaluate_rules(snapshot, samples=20)
    assert rules["all_songs"]["workflow"]["phrase_coverage"] == 1
    assert rules["all_songs"]["workflow-rag"]["phrase_coverage"] == 2 / 3
    assert rules["paired_songs"] == []
    assert build_tasks(snapshot, rules["paired_songs"])[0] == []


def test_separate_roots_are_recorded_and_mismatched_annotations_rejected(dataset, tmp_path):
    import shutil
    root, directory, _ = dataset
    other = tmp_path / "rag-copy"
    shutil.copytree(root, other)
    snapshot = collect(root, rag_root=other)
    assert snapshot["selection"]["roots"]["workflow-rag"] == str(other.resolve())
    assert snapshot["selection"]["roots"]["workflow"] == str(root.resolve())
    assert str(other) in snapshot["songs"][0]["candidates"]["workflow-rag"]["source"]["run"]
    path = other / directory.name / "song.yaml"
    changed = yaml.safe_load(path.read_text());changed["phrases"][0]["pattern"] = [2]
    path.write_text(yaml.safe_dump(changed, allow_unicode=True))
    with pytest.raises(ValueError, match="不能混合比较"):
        collect(root, rag_root=other)


def test_four_metric_report_and_rag_contrast_are_explicit(dataset, tmp_path):
    out = tmp_path / "report"
    assert main(["--root", str(dataset[0]), "--out", str(out), "--bootstrap-samples", "20"]) == 0
    result = read(out / "summary.json")
    assert result["variants"] == list(VARIANTS)
    assert result["metric_categories"] == ["syllable_constraints", "text_quality", "relative_quality", "comet_qe"]
    contrasts = result["rag_vs_workflow_rule_differences"]
    assert contrasts and all(r["a"] == "workflow-rag" and r["b"] == "workflow" for r in contrasts)
    report = (out / "report.md").read_text()
    for text in ("## 音节约束", "## 文本质量", "## 相对质量", "## 语义诊断"):
        assert text in report
    assert "## 实际可唱性" not in report


def test_parallel_judges_use_independent_clients_and_preserve_order(dataset, tmp_path):
    from threading import Lock
    snapshot = collect(dataset[0]);rules = evaluate_rules(snapshot, samples=20)
    clients = [];lock = Lock()
    class Judge:
        name = "parallel-fixture"
        def complete(self, *, agent, system, user):
            payload = json.loads(user)
            mapping = {r["label"]: "unused" for r in payload["candidates"]}
            task = {"mapping": mapping, "phrase_ids": [p["id"] for p in payload["phrases"]]}
            if len(mapping) == 4:
                result = score_result(task)
            else:
                result = {"winner": "tie", "reason": "无明显差异", "phrase_ids": [task["phrase_ids"][0]]}
            return json.dumps(result)
    def factory():
        judge = Judge()
        with lock:
            clients.append(judge)
        return judge
    result = run_judges(snapshot, rules, Judge(), tmp_path / "parallel", limit=1, jobs=4, client_factory=factory)
    assert result["completed_tasks"] == 13 and result["pending_tasks"] == 0
    assert len(clients) == 13 and len({id(c) for c in clients}) == 13
    assert all(p["win_rate_a"] == .5 for p in result["preferences"])
    # 降低并发续跑只读缓存，不重复调用客户端。
    cached = run_judges(snapshot, rules, Judge(), tmp_path / "parallel", limit=1)
    assert cached["completed_tasks"] == 13


def add_one_shot_fixture(directory, *, agent="writer", variant="one-shot"):
    song = Song.from_json(yaml.safe_load((directory / "song.yaml").read_text()))
    store = RunStore(directory / "baselines" / variant)
    run = store.new_run()
    for index, section in enumerate(song.sections, 1):
        store.prepare(run, agent, Request("fixture", "fixture", {
            "phrases": [p.to_json() for p in song.phrases], "focus": section.id,
        }), section_index_=index)
        store.finish(run, agent, {"section": section.id, "phrases": [
            {"id": p.id, "zh": "风 雨" if p.budget else "夜"}
            for p in song.translatable_of(section.id)
        ]}, meta={"model": "fixture-generator", "temperature": .7}, section_index_=index)
    store.set_latest(run, status="completed")
    return store, run


@pytest.mark.parametrize("agent", ["writer", "B"])
def test_one_shot_reader_validates_full_source_and_detects_missing_output(dataset, agent):
    root, directory, _ = dataset
    store, run = add_one_shot_fixture(directory, agent=agent)
    variants = (*VARIANTS, "one-shot")
    snapshot = collect(root, variants=variants)
    candidate = snapshot["songs"][0]["candidates"]["one-shot"]
    assert candidate["completed"] and candidate["texts"]["V1-03"] == "La"
    rules = evaluate_rules(snapshot, samples=20)
    assert rules["paired"]["one-shot"]["budget_pass_rate"] == 1
    request_path = store.record_dir(run, agent, 1) / "request.json"
    value = read(request_path)
    value["data"]["phrases"][0]["ja"] = "変更"
    write(request_path, value)
    with pytest.raises(ValueError, match="输入版本不同"):
        collect(root, variants=variants)
    value["data"]["phrases"][0]["ja"] = "あ"
    write(request_path, value)
    (store.record_dir(run, agent, 2) / "output.json").unlink()
    partial = evaluate_rules(collect(root, variants=variants), samples=20)
    assert partial["paired_songs"] == []
    assert partial["all_songs"]["one-shot"]["phrase_coverage"] == 2 / 3


def test_five_way_tasks_keep_existing_pair_requests_and_summarize_all_variants(dataset):
    root, directory, _ = dataset
    old_tasks, _ = build_tasks(collect(root), [directory.name])
    add_one_shot_fixture(directory)
    variants = (*VARIANTS, "one-shot")
    tasks, selection = build_tasks(collect(root, variants=variants), [directory.name])
    assert selection["planned_tasks"] == 42  # 2 sections × (1 score + 10 pairs × 2 orders)
    old_pairs = {t["id"]: t for t in old_tasks if t["kind"] == "pair"}
    new_pairs = {t["id"]: t for t in tasks if t["kind"] == "pair"}
    assert all(new_pairs[k] == value for k, value in old_pairs.items())
    outcomes = []
    for task in tasks:
        result = score_result(task) if task["kind"] == "score" else {
            "winner": "tie", "reason": "相同质量", "phrase_ids": task["phrase_ids"][:1],
        }
        outcomes.append({"status": "completed", "result": validate_result(result, task)})
    summary = summarize(tasks, outcomes, variants=variants)
    assert set(summary["absolute_paired"]) == set(variants)
    assert len(summary["preferences"]) == 10
    assert all(p["stable_pairs"] == 2 and p["win_rate_a"] == .5 for p in summary["preferences"])


def test_archived_writer_is_a_separate_sixth_candidate(dataset):
    root, directory, _ = dataset
    add_one_shot_fixture(directory)
    add_one_shot_fixture(directory, variant="one-shot-archive")
    variants = (*VARIANTS, "one-shot", "one-shot-archive")
    snapshot = collect(root, variants=variants)
    rules = evaluate_rules(snapshot, samples=20)
    assert set(rules["paired"]) == set(variants)
    assert rules["paired"]["one-shot-archive"]["budget_pass_rate"] == 1
    tasks, selection = build_tasks(snapshot, rules["paired_songs"])
    assert selection["planned_tasks"] == 62  # 2 × (1 + 15 × 2)
    assert len([t for t in tasks if t["kind"] == "score" and len(t["mapping"]) == 6]) == 2


def test_cli_can_compare_one_shot_with_workflow_only(dataset, tmp_path):
    root, directory, _ = dataset
    add_one_shot_fixture(directory)
    out = tmp_path / "ablation"
    assert main(["--root", str(root), "--variant", "workflow", "--variant", "one-shot",
                 "--out", str(out), "--bootstrap-samples", "20", "--judge-dry-run"]) == 0
    summary = read(out / "summary.json")
    assert summary["variants"] == ["workflow", "one-shot"]
    assert summary["judges"]["judge-1"]["task_count"] == 6
    assert "one-shot" in (out / "report.md").read_text()
    assert "paper-original" not in summary["paired"]


def test_five_way_reuses_only_identical_completed_pair_requests(dataset, tmp_path):
    root, directory, _ = dataset
    old_snapshot = collect(root)
    old_rules = evaluate_rules(old_snapshot, samples=20)
    old_tasks = build_tasks(old_snapshot, old_rules["paired_songs"], limit=1)[0]
    source = tmp_path / "old-judge"
    run_judges(old_snapshot, old_rules, ScriptedClient([json.dumps(result_for(t)) for t in old_tasks]), source, limit=1)
    add_one_shot_fixture(directory)
    variants = (*VARIANTS, "one-shot")
    snapshot = collect(root, variants=variants)
    rules = evaluate_rules(snapshot, samples=20)
    tasks = build_tasks(snapshot, rules["paired_songs"], limit=1)[0]
    existing_ids = {t["id"] for t in old_tasks}
    new_tasks = [t for t in tasks if t["id"] not in existing_ids]
    responses = [json.dumps(score_result(t) if t["kind"] == "score" else {
        "winner": "tie", "reason": "无明显差异", "phrase_ids": t["phrase_ids"][:1],
    }) for t in new_tasks]
    client = ScriptedClient(responses)
    out = tmp_path / "five-way"
    summary = run_judges(snapshot, rules, client, out, limit=1, reuse_from=source)
    assert summary["completed_tasks"] == 21
    assert len(client.calls) == 9  # 新打分 + one-shot 的四对顺序复评
    assert len(read(out / "reused-from.json")["task_ids"]) == 12
    assert len(summary["preferences"]) == 10
    bad_config = read(source / "config.json")
    bad_config["client"]["temperature"] = .9
    write(source / "config.json", bad_config)
    with pytest.raises(ValueError, match="client 设置不同"):
        run_judges(snapshot, rules, ScriptedClient([]), tmp_path / "bad", limit=1, reuse_from=source)
