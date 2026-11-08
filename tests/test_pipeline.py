"""整条流水线。

- ``workflow``：perceiver 一次 → 每段 writer → reviewer → 不合格带着 prev_zh 重来。
- ``rag``：perceiver 一次 → 每段 writer 出初版 → 用初版检索一次 → mimic 改写 →
  reviewer → 不合格只回 mimic 改写（参考复用）。

一次 ``run_song`` 产生一个 run 目录：perceiver 在 run 根，其余在 ``sections/NNN/``。
"""

import json

import pytest

from dubbing.llm import LLMError, NeedsLLM, PendingClient, ScriptedClient
from dubbing.models import LookupResult, PoetryHit
from dubbing.orchestrator import (
    RagError,
    attempt_section,
    plan_requests,
    run_song,
    store_for,
)
from dubbing.song import load_song
from dubbing.view import build_view

from helpers import analysis_json, draft_json, make_analysis, phrase, scripted, write_song


def _make(root):
    write_song(
        root,
        "t",
        ["S1", "S2"],
        [
            phrase("A-01", "S1", "ああああ", 4),
            phrase("A-02", "S1", "いいいい", 4),
            phrase("B-01", "S2", "ううう", 3),
        ],
    )


def _client(script):
    return ScriptedClient(script)


def _clean_client(root):
    return _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )


HIT = PoetryHit(
    chunk_id="顾城_gucheng/一代人#s1",
    text="黑夜给了我黑色的眼睛\n我却用它寻找光明",
    score=0.71,
    title="一代人",
    poet="顾城",
    poem_id="顾城_gucheng/一代人",
    stanza_idx=1,
)


class FakeRetriever:
    """替身：不装 chromadb，也不联网。"""

    k = 4
    collection = "fake_collection"
    index_id = "fake-v1"

    def __init__(self, hits=None, error: Exception | None = None):
        self.hits = list(hits or [])
        self.error = error
        self.queries: list[str] = []
        self.excludes: list[set[str] | None] = []

    def search(self, query_text, *, k=None, exclude=None):
        self.queries.append(query_text)
        # 快照：编排器传的是同一个可变集合，晚一点再读就看不到当时的样子了
        self.excludes.append(set(exclude) if exclude is not None else None)
        if self.error is not None:
            raise self.error
        return list(self.hits)


# -- workflow 管线 -------------------------------------------------------


def test_happy_path_with_one_repair(root):
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),  # A-01 超 1 字
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),  # 修好
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )

    outcome = run_song("t", client, root=root)

    assert outcome.ok
    assert outcome.pipeline == "workflow"
    assert outcome.run_id and outcome.run_id.startswith("run-")
    s1, s2 = outcome.sections
    assert (s1.section, s1.status, s1.rounds) == ("S1", "passed", 2)
    assert (s2.section, s2.status, s2.rounds) == ("S2", "passed", 1)

    # perceiver 一次 + S1 两次 + S2 一次
    assert [c["agent"] for c in client.calls] == ["perceiver", "writer", "writer", "writer"]


def test_one_run_directory_per_song(root):
    _make(root)
    outcome = run_song("t", _clean_client(root), root=root)

    store = store_for(root, "t")
    assert store.list_runs() == [outcome.run_id]
    assert store.latest_run() == outcome.run_id
    run_dir = store.run_dir(outcome.run_id)
    assert (run_dir / "perceiver" / "output.json").is_file()
    assert (run_dir / "sections" / "001" / "writer" / "output.json").is_file()
    assert (run_dir / "sections" / "001" / "reviewer" / "output.json").is_file()
    assert (run_dir / "sections" / "002" / "writer" / "output.json").is_file()
    assert (run_dir / "lyrics.zh.txt").is_file() and (run_dir / "lyrics.md").is_file()
    # 多轮返修只留最终一轮
    assert store.read_meta(outcome.run_id, "writer", 1)["round"] == 1
    assert not (run_dir / "sections" / "001" / "writer" / "response.txt").exists()
    assert store.read_meta(outcome.run_id, "writer", 1)["raw_response"]


def test_repair_round_carries_prev_zh(root):
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)

    second_writer = client.calls[2]
    data = json.loads(second_writer["user"].split("```json")[1].split("```")[0])
    rows = {r["id"]: r for r in data["phrases"]}
    assert rows["A-01"]["prev_zh"] == "一二三四五"
    assert "zh" not in rows["A-01"]


def test_repair_round_tells_writer_what_was_wrong(root):
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)

    second_writer = client.calls[2]["user"]
    assert "上一轮" in second_writer
    assert "A-01" in second_writer
    assert "超出 1 字" in second_writer
    assert "[必修]" in second_writer


def test_first_round_has_no_feedback(root):
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)
    assert "上一轮" not in client.calls[1]["user"]


def test_finished_section_is_finalized_for_later_sections(root):
    _make(root)
    client = _clean_client(root)
    run_song("t", client, root=root)

    s2_call = json.loads(client.calls[2]["user"].split("```json")[1].split("```")[0])
    rows = {r["id"]: r for r in s2_call["phrases"]}
    assert rows["A-01"]["zh"] == "一二三四"  # 已定稿，作为只读参考
    assert "prev_zh" not in rows["A-01"]


def test_exhausted_and_regression_recorded(root):
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                # 第一轮：A-01 超字
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),
                # 第二轮：修好了 A-01，却把 A-02 改坏
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九十"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )

    outcome = run_song("t", client, root=root, max_rounds=2)

    assert not outcome.ok
    s1 = outcome.sections[0]
    assert (s1.status, s1.rounds) == ("exhausted", 2)
    assert s1.regressions == ("A-02",)


def test_runs_and_records_are_on_disk(root):
    _make(root)
    outcome = run_song("t", _clean_client(root), root=root)

    store = store_for(root, "t")
    song = load_song("t", root=root)
    view = build_view(song, store, outcome.run_id)
    assert {r.phrase_id: r.verdict for r in view.rows} == {
        "A-01": "pass",
        "A-02": "pass",
        "B-01": "pass",
    }
    assert view.row("A-01").sing_text == "一二三四"

    meta = store.read_meta(outcome.run_id, "writer", 1)
    assert meta["agent"] == "writer"
    assert meta["prompt_hash"]
    assert meta["model"]


def test_pending_client_stops_at_first_call_and_keeps_request(root):
    _make(root)
    with pytest.raises(NeedsLLM, match="request.json"):
        run_song("t", PendingClient(), root=root)

    store = store_for(root, "t")
    runs = store.list_runs()
    assert len(runs) == 1
    payload = store.read_request(runs[0], "perceiver")
    assert "system" in payload["messages"]
    assert payload["data"]["sections"][0]["lyrics"] == ["ああああ", "いいいい"]
    assert "abandoned" in store.read_meta(runs[0], "perceiver")


def test_plan_requests_previews_everything_without_calling(root):
    _make(root)
    plan = plan_requests("t", root=root)
    assert [name for name, _ in plan] == ["perceiver", "writer-S1", "writer-S2"]
    assert not (root / "t" / "workflow").exists()


def test_second_run_is_a_new_directory_and_reuses_analysis(root):
    _make(root)
    first = run_song("t", _clean_client(root), root=root)

    second = _client(
        {
            "perceiver": [],
            "writer": scripted(
                draft_json("S1", {"A-01": "甲乙丙丁", "A-02": "戊己庚辛"}),
                draft_json("S2", {"B-01": "丑寅卯"}),
            ),
        }
    )
    outcome = run_song("t", second, root=root)

    assert outcome.analysis_reused
    assert outcome.run_id != first.run_id
    assert [s.status for s in outcome.sections] == ["passed", "passed"]
    assert [c["agent"] for c in second.calls] == ["writer", "writer"]
    # 从头生成：不给上一版，reviewer 也不跟旧版比较
    assert "prev_zh" not in second.calls[0]["user"]

    store = store_for(root, "t")
    assert store.list_runs() == [first.run_id, outcome.run_id]
    assert store.latest_run() == outcome.run_id
    # perceiver 是整份抄过来的，没有重复花钱
    assert store.has_output(outcome.run_id, "perceiver")


def test_continue_feeds_previous_translation(root):
    _make(root)
    run_song("t", _clean_client(root), root=root)

    second = _client(
        {
            "perceiver": [],
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", second, root=root, fresh=False)

    assert "prev_zh" in second.calls[0]["user"]
    assert "一二三四" in second.calls[0]["user"]


def test_fresh_round_one_reports_no_regressions(root):
    """从头生成时没有“上一轮”，changes 全是 rewritten，regressions 为空。"""
    _make(root)
    run_song("t", _clean_client(root), root=root)

    second = _client(
        {
            "perceiver": [],
            "writer": scripted(
                draft_json("S1", {"A-01": "五六七八", "A-02": "九十甲乙"}),
                draft_json("S2", {"B-01": "丙丁戊"}),
            ),
        }
    )
    outcome = run_song("t", second, root=root)

    store = store_for(root, "t")
    review = store.read_output(outcome.run_id, "reviewer", 1)
    assert review["changes"]["unchanged"] == []
    assert review["regressions"] == []


def test_skip_done_copies_finished_sections_forward(root):
    _make(root)
    first = run_song("t", _clean_client(root), root=root)

    second = _client({"perceiver": [], "writer": []})
    outcome = run_song("t", second, root=root, skip_done=True)

    assert [s.status for s in outcome.sections] == ["reused", "reused"]
    assert second.calls == []
    assert outcome.run_id != first.run_id

    store = store_for(root, "t")
    assert store.read_output(outcome.run_id, "writer", 1) == store.read_output(first.run_id, "writer", 1)
    assert store.has_output(outcome.run_id, "reviewer", 2)


class BoomClient:
    """每次调用都失败，用来验证失败也留下痕迹。"""

    name = "boom"

    def complete(self, *, agent, system, user):
        raise LLMError("HTTP 400：Model Not Exist")


def test_failed_call_marks_run_abandoned(root):
    _make(root)
    with pytest.raises(LLMError, match="request.json"):
        run_song("t", BoomClient(), root=root)

    store = store_for(root, "t")
    run_ids = store.list_runs()
    assert len(run_ids) == 1
    assert "abandoned" in store.read_meta(run_ids[0], "perceiver")


def test_only_runs_the_selected_section(root):
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(draft_json("S2", {"B-01": "一二三"})),
        }
    )
    outcome = run_song("t", client, root=root, only={"S2"})

    assert [s.section for s in outcome.sections] == ["S2"]
    assert [c["agent"] for c in client.calls] == ["perceiver", "writer"]

    store = store_for(root, "t")
    assert store.has_output(outcome.run_id, "writer", 2)
    assert not store.has_output(outcome.run_id, "writer", 1)


def test_only_carries_the_other_sections_over(root):
    _make(root)
    first = run_song("t", _clean_client(root), root=root)

    client = _client({"perceiver": [], "writer": scripted(draft_json("S2", {"B-01": "丙丁戊"}))})
    outcome = run_song("t", client, root=root, only={"S2"})

    store = store_for(root, "t")
    assert [c["agent"] for c in client.calls] == ["writer"]
    assert store.read_output(outcome.run_id, "writer", 1) == store.read_output(first.run_id, "writer", 1)
    assert store.read_output(outcome.run_id, "writer", 2)["phrases"][0]["zh"] == "丙丁戊"


def test_only_rejects_unknown_section(root):
    _make(root)
    with pytest.raises(KeyError, match="没有这些 section"):
        run_song("t", PendingClient(), root=root, only={"S9"})


# -- 进度事件 ------------------------------------------------------------


def test_emits_one_step_pair_per_call(root):
    _make(root)
    steps = []
    run_song("t", _clean_client(root), root=root, on_step=steps.append)

    shape = [(s.agent, s.phase, s.section) for s in steps]
    assert shape == [
        ("perceiver", "start", None),
        ("perceiver", "done", None),
        ("writer", "start", "S1"),
        ("reviewer", "done", "S1"),
        ("writer", "start", "S2"),
        ("reviewer", "done", "S2"),
    ]
    assert steps[3].note.startswith("pass")


def test_steps_show_each_repair_round(root):
    _make(root)
    steps = []
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root, on_step=steps.append)

    s1 = [s for s in steps if s.section == "S1"]
    assert [s.note for s in s1 if s.phase == "start"] == ["第 1 轮", "第 2 轮"]
    assert s1[1].note.startswith("fail")


def test_stale_analysis_is_flagged_when_song_gained_a_section(root):
    """改 song.yaml 加了段之后，perceiver 的旧产物不会自动失效——必须提醒。"""
    write_song(root, "t", ["S1"], [phrase("A-01", "S1", "ああああ", 4)])
    first = _client(
        {
            "perceiver": scripted(analysis_json(["S1"])),
            "writer": scripted(draft_json("S1", {"A-01": "一二三四"})),
        }
    )
    run_song("t", first, root=root)

    # 人工加一段 S2
    write_song(
        root,
        "t",
        ["S1", "S2"],
        [phrase("A-01", "S1", "ああああ", 4), phrase("B-01", "S2", "いいい", 3)],
    )
    second = _client({"perceiver": [], "writer": scripted(draft_json("S2", {"B-01": "一二三"}))})
    outcome = run_song("t", second, root=root, only={"S2"})

    assert outcome.analysis_reused is True
    assert second.calls and second.calls[0]["agent"] == "writer"  # perceiver 没重跑
    assert any("没有覆盖" in w and "S2" in w for w in outcome.warnings)


def test_broken_json_keeps_the_raw_response(root):
    """连 json_repair 都救不回来时，必须留下模型原话，否则没法复盘。"""
    _make(root)
    broken = "抱歉，这一段我写不出来。"
    client = _client({"perceiver": scripted(analysis_json(["S1", "S2"])), "writer": [broken]})

    with pytest.raises(LLMError, match="找不到 JSON"):
        run_song("t", client, root=root)

    store = store_for(root, "t")
    meta = store.read_meta(store.list_runs()[0], "writer", 1)
    assert meta["raw_response"] == broken
    assert "解析失败" in meta["abandoned"]


def test_raw_newline_in_string_does_not_break_the_run(root):
    """同样的换行，现在应当被容忍而不是整轮失败。"""
    _make(root)
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": [
                '{"section": "S1", "phrases": [{"id": "A-01", "zh": "一二\n三四"}, '
                '{"id": "A-02", "zh": "六七八九"}]}',
                json.dumps(draft_json("S2", {"B-01": "一二三"}), ensure_ascii=False),
            ],
        }
    )
    outcome = run_song("t", client, root=root)
    assert outcome.sections[0].status == "passed"


# -- rag 管线 ------------------------------------------------------------


def _rag_client(*, perceiver=None, writer=None, mimic=None):
    payload: dict[str, list[str]] = {}
    if perceiver is not None:
        payload["perceiver"] = scripted(*perceiver)
    if writer is not None:
        payload["writer"] = scripted(*writer)
    if mimic is not None:
        payload["mimic"] = scripted(*mimic)
    return ScriptedClient(payload)


def test_rag_writes_lookup_write_and_reviewer_records(root):
    _make(root)
    retriever = FakeRetriever([HIT])
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"})],
        mimic=[draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"})],
    )

    outcome = run_song("t", client, root=root, pipeline="rag", retriever=retriever, only={"S1"})

    assert outcome.pipeline == "rag"
    assert outcome.ok
    store = store_for(root, "t", "rag")
    run_dir = store.run_dir(outcome.run_id)
    assert (run_dir / "sections" / "001" / "writer" / "output.json").is_file()
    assert (run_dir / "sections" / "001" / "mimic_lookup" / "output.json").is_file()
    assert (run_dir / "sections" / "001" / "mimic_write" / "output.json").is_file()
    assert (run_dir / "sections" / "001" / "reviewer" / "output.json").is_file()

    # 查询文本就是 writer 的初版译文
    assert retriever.queries == ["一二三四\n六七八九"]
    lookup = store.read_output(outcome.run_id, "mimic_lookup", 1)
    assert lookup["query_text"] == "一二三四\n六七八九"
    assert lookup["hits"][0]["chunk_id"] == HIT.chunk_id
    assert lookup["index_id"] == "fake-v1"
    assert lookup["skipped"] == ""

    # mimic 的入参里既有整曲原文，也有参考诗句与初版
    mimic_call = json.loads(client.calls[-1]["user"].split("```json")[1].split("```")[0])
    assert mimic_call["focus"] == "S1"
    assert mimic_call["draft"]["phrases"][0]["zh"] == "一二三四"
    assert mimic_call["references"][0]["text"].startswith("黑夜给了我")
    assert mimic_call["phrases"][-1]["section"] == "S2"

    # 最终文本取自 mimic_write
    song = load_song("t", root=root)
    view = build_view(song, store, outcome.run_id, pipeline="rag")
    assert view.row("A-01").sing_text == "一二三四"

    # 每条记录都记了索引版本，方便日后解释差异
    assert store.read_meta(outcome.run_id, "mimic_write", 1)["index_id"] == "fake-v1"


def test_rag_repair_goes_back_to_mimic_and_reuses_references(root):
    _make(root)
    retriever = FakeRetriever([HIT])
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"})],
        mimic=[
            draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),  # 超 1 字
            draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),  # 改好
        ],
    )

    outcome = run_song("t", client, root=root, pipeline="rag", retriever=retriever, only={"S1"})

    assert [c["agent"] for c in client.calls] == ["perceiver", "writer", "mimic", "mimic"]
    assert len(retriever.queries) == 1  # 返修不重新检索
    assert outcome.sections[0].rounds == 2

    second_mimic = json.loads(client.calls[-1]["user"].split("```json")[1].split("```")[0])
    assert second_mimic["references"][0]["chunk_id"] == HIT.chunk_id
    assert second_mimic["draft"]["phrases"][0]["zh"] == "一二三四"  # 初版没变
    assert "上一轮" in client.calls[-1]["user"]


def test_rag_skips_lookup_when_the_draft_is_empty(root):
    _make(root)
    retriever = FakeRetriever([HIT])
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[draft_json("S1", {})],  # writer 什么都没写
        mimic=[draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"})],
    )

    outcome = run_song("t", client, root=root, pipeline="rag", retriever=retriever, only={"S1"})

    assert retriever.queries == []  # 没有东西可查，不去打扰索引
    store = store_for(root, "t", "rag")
    lookup = store.read_output(outcome.run_id, "mimic_lookup", 1)
    assert lookup["skipped"] == "empty-query"
    assert lookup["hits"] == []
    assert outcome.ok  # 没有参考也要能跑完

    mimic_call = json.loads(client.calls[-1]["user"].split("```json")[1].split("```")[0])
    assert mimic_call["references"] == []


def test_rag_requires_a_retriever(root):
    _make(root)
    song = load_song("t", root=root)
    store = store_for(root, "t", "rag")
    run_id = store.new_run()
    client = _rag_client(writer=[draft_json("S1", {"A-01": "一二三四"})])

    with pytest.raises(RagError, match="需要检索器"):
        attempt_section(
            song,
            store,
            run_id,
            None,
            make_analysis(["S1", "S2"]),
            client,
            "S1",
            pipeline="rag",
            retriever=None,
        )


def test_rag_retrieval_failure_is_recorded_and_raised(root):
    """基础设施故障不许静默退化成"没有参考"。"""
    _make(root)
    retriever = FakeRetriever(error=RuntimeError("embedding service down"))
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"})],
    )

    with pytest.raises(RagError, match="检索失败"):
        run_song("t", client, root=root, pipeline="rag", retriever=retriever, only={"S1"})

    store = store_for(root, "t", "rag")
    meta = store.read_meta(store.list_runs()[0], "mimic_lookup", 1)
    assert "embedding service down" in meta["abandoned"]


def test_rag_writer_retry_is_opt_in(root):
    _make(root)
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[
            draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
            draft_json("S1", {"A-01": "甲乙丙丁", "A-02": "戊己庚辛"}),
        ],
        mimic=[
            draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),  # 第一轮初版：不过
            draft_json("S1", {"A-01": "一二三四五六", "A-02": "六七八九"}),  # 第二轮初版：也不过
        ],
    )
    retriever = FakeRetriever([HIT])

    outcome = run_song(
        "t",
        client,
        root=root,
        pipeline="rag",
        retriever=retriever,
        only={"S1"},
        max_rounds=1,
        writer_retry=True,
    )

    assert not outcome.ok
    assert [c["agent"] for c in client.calls] == ["perceiver", "writer", "mimic", "writer", "mimic"]
    assert len(retriever.queries) == 1  # 重出初版也不重新检索
    # 第二次初版带着上一轮评审意见
    assert "上一轮" in client.calls[3]["user"]


def test_rag_dedupe_peers_excludes_chunks_used_by_earlier_sections(root):
    _make(root)
    retriever = FakeRetriever([HIT])
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[
            draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
            draft_json("S2", {"B-01": "一二三"}),
        ],
        mimic=[
            draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
            draft_json("S2", {"B-01": "一二三"}),
        ],
    )

    run_song("t", client, root=root, pipeline="rag", retriever=retriever, dedupe_peers=True)

    assert retriever.excludes[0] is None or HIT.chunk_id not in (retriever.excludes[0] or set())
    assert HIT.chunk_id in retriever.excludes[1]


def test_plan_requests_for_rag_previews_writer_and_mimic(root):
    _make(root)
    plan = plan_requests("t", root=root, pipeline="rag")
    assert [name for name, _ in plan] == [
        "perceiver",
        "writer-S1",
        "mimic-S1",
        "writer-S2",
        "mimic-S2",
    ]
    assert plan[2][1].data["references"] == []


def test_pipelines_keep_separate_run_directories(root):
    _make(root)
    run_song("t", _clean_client(root), root=root)
    assert (root / "t" / "workflow" / "latest.json").is_file()
    assert not (root / "t" / "workflow_rag").exists()

    retriever = FakeRetriever([HIT])
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[
            draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
            draft_json("S2", {"B-01": "一二三"}),
        ],
        mimic=[
            draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
            draft_json("S2", {"B-01": "一二三"}),
        ],
    )
    run_song("t", client, root=root, pipeline="rag", retriever=retriever)

    assert (root / "t" / "workflow_rag" / "latest.json").is_file()
    assert len(store_for(root, "t", "workflow").list_runs()) == 1
    assert len(store_for(root, "t", "rag").list_runs()) == 1


def test_mimic_output_is_reviewed_and_can_be_failed(root):
    """reviewer 拿到的是 mimic 的产物，不合格就回 mimic。"""
    _make(root)
    retriever = FakeRetriever([HIT])
    client = _rag_client(
        perceiver=[analysis_json(["S1", "S2"])],
        writer=[draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"})],
        mimic=[draft_json("S1", {"A-01": "一二三四五六", "A-02": "六七八九"})],
    )

    outcome = run_song(
        "t", client, root=root, pipeline="rag", retriever=retriever, only={"S1"}, max_rounds=1
    )

    assert outcome.sections[0].status == "exhausted"
    store = store_for(root, "t", "rag")
    review = store.read_output(outcome.run_id, "reviewer", 1)
    assert review["verdict"] == "fail"
    assert review["draft_run"] == outcome.run_id


def test_lookup_result_round_trips_through_json():
    result = LookupResult(section="S1", query_text="q", k=4, hits=(HIT,), index_id="idx")
    again = LookupResult.from_json(result.to_json())
    assert again == result


def test_repaired_json_is_recorded_in_meta(root):
    """模型输出结构手滑（错括号）时用 json_repair 兜底，并在 meta 里留痕。"""
    _make(root)
    broken = (
        '{"section": "S1", "phrases": ['
        '{"id": "A-01", "zh": "一二三四", "chars": 4},'
        '{"id": "A-02", "zh": "六七八九", "self_check": '
        '{"budget": [4, 4], "ok": true, "note": "四字整"]}'
        "]}"
    )
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            # 坏的那条要原样塞进去，不能过 scripted()——它会把字符串 json.dumps 成合法 JSON 字符串
            "writer": [broken, *scripted(draft_json("S2", {"B-01": "一二三"}))],
        }
    )

    outcome = run_song("t", client, root=root)

    assert outcome.ok  # 修好之后这一轮就过了，不必重来
    store = store_for(root, "t")
    assert store.read_meta(outcome.run_id, "writer", 1)["json_repaired"] is True
    assert store.read_meta(outcome.run_id, "writer", 2)["json_repaired"] is False
    assert store.read_output(outcome.run_id, "writer", 1)["phrases"][1]["zh"] == "六七八九"


def test_truncated_output_becomes_an_incomplete_draft_and_fails_review(root):
    """截断的输出被修成不完整草稿：由 reviewer 报 UNDER_BUDGET，而不是整轮崩掉。"""
    _make(root)
    truncated = (
        '{"section": "S1", "phrases": [{"id": "A-01", "zh": "一二三四", "chars": 4}, '
        '{"id": "A-02", "zh": "六'
    )
    client = _client(
        {
            "perceiver": scripted(analysis_json(["S1", "S2"])),
            "writer": [truncated],
        }
    )

    outcome = run_song("t", client, root=root, only={"S1"}, max_rounds=1)

    assert outcome.sections[0].status == "exhausted"
    store = store_for(root, "t")
    assert store.read_meta(outcome.run_id, "writer", 1)["json_repaired"] is True
    review = store.read_output(outcome.run_id, "reviewer", 1)
    assert review["verdict"] == "fail"
    assert any(v["code"] == "UNDER_BUDGET" for v in review["violations"])
