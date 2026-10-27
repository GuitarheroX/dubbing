"""人看的渲染：每个 agent 的产物长什么样，以及 runs / show 的数据来源。"""

from dubbing.models import Analysis, Draft, Review
from dubbing.orchestrator import run_song
from dubbing.llm import ScriptedClient
from dubbing.report import (
    current_run_ids,
    list_runs,
    render_analysis,
    render_draft,
    render_review,
    render_run,
)
from dubbing.song import load_song
from dubbing.store import SongStore

from helpers import analysis_json, draft_json, make_song, phrase, scripted, write_song

SONG = make_song(
    "t", ["S1"], [phrase("P1", "S1", "あああ", 3), phrase("P2", "S1", "いいい", 3)]
)


# -- 单个产物的渲染 ------------------------------------------------------


def test_render_analysis_contains_every_part():
    text = render_analysis("r1", Analysis.from_json(analysis_json(["S1"])))
    assert "主题" in text
    assert "情绪走向" in text
    assert "末班车的尾灯" in text
    assert "S1：" in text
    assert "独白的散文" in text


def test_render_draft_shows_counts_and_self_check():
    draft = Draft.from_json(
        {
            "section": "S1",
            "phrases": [
                {"id": "P1", "zh": "一二三", "self_check": {"note": "三字整"}},
                {"id": "P2", "zh": "四五六"},
            ],
        }
    )
    text = render_draft("r2", SONG, draft)
    assert "一二三" in text and "四五六" in text
    assert "↳ 三字整" in text
    assert "3-3" in text


def test_render_review_lists_violations_and_changes():
    review = Review.from_json(
        {
            "section": "S1",
            "draft_run": "r2",
            "verdict": "fail",
            "violations": [
                {
                    "phrase_id": "P1",
                    "code": "OVER_BUDGET",
                    "severity": "hard",
                    "expected": [3, 3],
                    "actual": 5,
                    "fix": "超出 2 字，压到 3 字以内",
                }
            ],
            "changes": {"rewritten": ["P1"], "unchanged": ["P2"]},
            "regressions": ["P1"],
        }
    )
    text = render_review("r3", review)
    assert "fail" in text
    assert "OVER_BUDGET" in text and "超出 2 字" in text
    assert "改写过：P1" in text
    assert "未动过：P2" in text
    assert "回归" in text and "P1" in text


# -- runs / show ---------------------------------------------------------


def _run_pipeline(root):
    write_song(
        root,
        "t",
        ["S1"],
        [phrase("P1", "S1", "あああ", 3), phrase("P2", "S1", "いいい", 3)],
    )
    client = ScriptedClient(
        {
            "a": scripted(analysis_json(["S1"])),
            "b": scripted(draft_json("S1", {"P1": "一二三", "P2": "四五六"})),
        }
    )
    run_song("t", client, root=root)
    return SongStore(root, "t")


def test_list_runs_marks_the_current_ones(root):
    store = _run_pipeline(root)
    state = store.load_state()
    rows = list_runs(store, state)

    assert {r["agent"] for r in rows} == {"a", "b", "c"}
    current = [r for r in rows if r["current"]]
    assert {r["agent"] for r in current} == {"a", "b", "c"}
    assert current_run_ids(state) == {r["run_id"] for r in current}

    b_row = next(r for r in rows if r["agent"] == "b")
    assert b_row["note"] == "2 句"
    c_row = next(r for r in rows if r["agent"] == "c")
    assert c_row["note"] == "pass"


def test_list_runs_survives_an_interrupted_run(root):
    store = _run_pipeline(root)
    broken = store.runs_dir / "20260101T000000_b_abcdef"
    broken.mkdir()
    (broken / "request.json").write_text("{}", encoding="utf-8")

    rows = list_runs(store, store.load_state())
    busy = next(r for r in rows if r["run_id"] == broken.name)
    assert busy["agent"] == "b"
    assert "未完成" in busy["note"]


def test_render_run_dispatches_by_agent(root):
    store = _run_pipeline(root)
    state = store.load_state()
    song = load_song("t", root=root)

    assert render_run(store, song, state.analysis).startswith("A 的体会")
    assert "B 的译词" in render_run(store, song, state.sections["S1"].draft)
    assert "C 的检查" in render_run(store, song, state.sections["S1"].review)
