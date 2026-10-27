"""整条流水线：A 一次 → 每段 B → C → 不合格带着 prev_zh 重来。"""

import json

import pytest

from dubbing.llm import LLMError, NeedsLLM, PendingClient, ScriptedClient
from dubbing.orchestrator import plan_requests, run_song
from dubbing.song import load_song
from dubbing.store import SongStore
from dubbing.view import build_view

from helpers import analysis_json, draft_json, phrase, scripted, write_song


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


def test_happy_path_with_one_repair(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),  # A-01 超 1 字
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),  # 修好
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )

    outcome = run_song("t", client, root=root)

    assert outcome.ok
    s1, s2 = outcome.sections
    assert (s1.section, s1.status, s1.rounds) == ("S1", "passed", 2)
    assert (s2.section, s2.status, s2.rounds) == ("S2", "passed", 1)

    # A 一次 + S1 两次 + S2 一次
    assert [c["agent"] for c in client.calls] == ["a", "b", "b", "b"]


def test_repair_round_carries_prev_zh(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)

    second_b = client.calls[2]
    data = json.loads(second_b["user"].split("```json")[1].split("```")[0])
    rows = {r["id"]: r for r in data["phrases"]}
    assert rows["A-01"]["prev_zh"] == "一二三四五"
    assert "zh" not in rows["A-01"]


def test_repair_round_tells_b_what_was_wrong(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四五", "A-02": "六七八九"}),
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)

    second_b = client.calls[2]["user"]
    assert "上一轮" in second_b
    assert "A-01" in second_b
    assert "超出 1 字" in second_b
    assert "[必修]" in second_b


def test_first_round_has_no_feedback(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)
    assert "上一轮" not in client.calls[1]["user"]


def test_finished_section_is_finalized_for_later_sections(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)

    s2_call = json.loads(client.calls[2]["user"].split("```json")[1].split("```")[0])
    rows = {r["id"]: r for r in s2_call["phrases"]}
    assert rows["A-01"]["zh"] == "一二三四"  # 已定稿，作为只读参考
    assert "prev_zh" not in rows["A-01"]


def test_exhausted_and_regression_recorded(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
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


def test_state_and_runs_are_on_disk(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root)

    store = SongStore(root, "t")
    state = store.load_state()
    assert state.analysis
    assert state.sections["S1"].verdict == "pass"
    assert state.sections["S2"].verdict == "pass"
    assert store.has_run(state.sections["S1"].draft)

    meta = store.load_meta(state.sections["S1"].draft)
    assert meta["agent"] == "b"
    assert meta["prompt_hash"]
    assert meta["input_hash"]

    view = build_view(load_song("t", root=root), state, store)
    assert {r.phrase_id: r.verdict for r in view.rows} == {
        "A-01": "pass",
        "A-02": "pass",
        "B-01": "pass",
    }
    assert view.row("A-01").sing_text == "一二三四"


def test_pending_client_stops_at_first_call_and_keeps_request(root):
    _make(root)
    with pytest.raises(NeedsLLM, match="request.json"):
        run_song("t", PendingClient(), root=root)

    runs = root / "t" / "runs"
    requests = list(runs.glob("*/request.json"))
    assert len(requests) == 1
    payload = json.loads(requests[0].read_text(encoding="utf-8"))
    assert "system" in payload["messages"]
    assert payload["data"]["sections"][0]["lyrics"] == ["ああああ", "いいいい"]


def test_plan_requests_previews_everything_without_calling(root):
    _make(root)
    plan = plan_requests("t", root=root)
    assert [name for name, _ in plan] == ["a", "b:S1", "b:S2"]
    assert not (root / "t" / "runs").exists()


def test_second_run_reuses_analysis_but_re_rolls_sections(root):
    _make(root)
    first = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", first, root=root)

    # 默认行为：忽略现有译文，从零重生成并覆盖（A 复用，不重复花钱）
    second = _client(
        {
            "a": [],
            "b": scripted(
                draft_json("S1", {"A-01": "甲乙丙丁", "A-02": "戊己庚辛"}),
                draft_json("S2", {"B-01": "丑寅卯"}),
            ),
        }
    )
    outcome = run_song("t", second, root=root)

    assert outcome.analysis_reused
    assert [s.status for s in outcome.sections] == ["passed", "passed"]
    assert [c["agent"] for c in second.calls] == ["b", "b"]
    # 从头生成：不给上一版，C 也不跟旧版比较
    assert "prev_zh" not in second.calls[0]["user"]


def test_continue_feeds_previous_translation(root):
    _make(root)
    first = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", first, root=root)

    second = _client(
        {
            "a": [],
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", second, root=root, fresh=False)

    assert "prev_zh" in second.calls[0]["user"]
    assert "一二三四" in second.calls[0]["user"]


def test_fresh_round_one_reports_no_regressions(root):
    """从头生成时没有"上一轮"，changes 全是 rewritten，regressions 为空。"""
    _make(root)
    first = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", first, root=root)

    second = _client(
        {
            "a": [],
            "b": scripted(
                draft_json("S1", {"A-01": "五六七八", "A-02": "九十甲乙"}),
                draft_json("S2", {"B-01": "丙丁戊"}),
            ),
        }
    )
    run_song("t", second, root=root)

    store = SongStore(root, "t")
    state = store.load_state()
    review = store.load_output(state.sections["S1"].review)
    assert review["changes"]["unchanged"] == []
    assert review["regressions"] == []


def test_skip_done_leaves_finished_sections_alone(root):
    _make(root)
    first = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", first, root=root)

    second = _client({"a": [], "b": []})
    outcome = run_song("t", second, root=root, skip_done=True)

    assert [s.status for s in outcome.sections] == ["reused", "reused"]
    assert second.calls == []


class BoomClient:
    """每次调用都失败，用来验证失败也留下痕迹。"""

    name = "boom"

    def complete(self, *, agent, system, user):
        raise LLMError("HTTP 400：Model Not Exist")


def test_failed_call_marks_run_abandoned(root):
    _make(root)
    with pytest.raises(LLMError, match="request.json"):
        run_song("t", BoomClient(), root=root)

    metas = list((root / "t" / "runs").glob("*/meta.json"))
    assert len(metas) == 1
    assert "abandoned" in json.loads(metas[0].read_text(encoding="utf-8"))


def test_only_runs_the_selected_section(root):
    _make(root)
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(draft_json("S2", {"B-01": "一二三"})),
        }
    )
    outcome = run_song("t", client, root=root, only={"S2"})

    assert [s.section for s in outcome.sections] == ["S2"]
    assert [c["agent"] for c in client.calls] == ["a", "b"]

    store = SongStore(root, "t")
    assert set(store.load_state().sections) == {"S2"}


def test_only_rejects_unknown_section(root):
    _make(root)
    with pytest.raises(KeyError, match="没有这些 section"):
        run_song("t", PendingClient(), root=root, only={"S9"})


# -- 进度事件 ------------------------------------------------------------


def test_emits_one_step_pair_per_call(root):
    _make(root)
    steps = []
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
                draft_json("S1", {"A-01": "一二三四", "A-02": "六七八九"}),
                draft_json("S2", {"B-01": "一二三"}),
            ),
        }
    )
    run_song("t", client, root=root, on_step=steps.append)

    shape = [(s.agent, s.phase, s.section) for s in steps]
    assert shape == [
        ("a", "start", None),
        ("a", "done", None),
        ("b", "start", "S1"),
        ("c", "done", "S1"),
        ("b", "start", "S2"),
        ("c", "done", "S2"),
    ]
    assert steps[3].note.startswith("pass")


def test_steps_show_each_repair_round(root):
    _make(root)
    steps = []
    client = _client(
        {
            "a": scripted(analysis_json(["S1", "S2"])),
            "b": scripted(
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
    """改 song.yaml 加了段之后，A 的旧产物不会自动失效——必须提醒。"""
    write_song(
        root,
        "t",
        ["S1"],
        [phrase("A-01", "S1", "ああああ", 4)],
    )
    first = _client(
        {
            "a": scripted(analysis_json(["S1"])),
            "b": scripted(draft_json("S1", {"A-01": "一二三四"})),
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
    second = _client({"a": [], "b": scripted(draft_json("S2", {"B-01": "一二三"}))})
    outcome = run_song("t", second, root=root, only={"S2"})

    assert outcome.analysis_reused is True
    assert second.calls and second.calls[0]["agent"] == "b"  # A 没重跑
    assert any("没有覆盖" in w and "S2" in w for w in outcome.warnings)
