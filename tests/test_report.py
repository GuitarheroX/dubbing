"""人看的渲染：每个 agent 的产物长什么样，以及 runs / show 的数据来源。"""

from dubbing.llm import ScriptedClient
from dubbing.models import Analysis, Draft, Review
from dubbing.orchestrator import run_song, workflow_store
from dubbing.report import (
    list_runs,
    render_analysis,
    render_draft,
    render_review,
    render_record,
)
from dubbing.song import load_song

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
    outcome = run_song("t", client, root=root)
    return workflow_store(root, "t"), load_song("t", root=root), outcome


def test_list_runs_marks_the_current_one(root):
    store, song, outcome = _run_pipeline(root)
    rows = list_runs(store, song)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == outcome.run_id
    assert row["current"] is True
    assert row["model"] == "scripted"
    assert row["has_a"] is True
    assert (row["passed"], row["total"]) == (1, 1)
    assert row["status"] == "completed"
    assert row["lyrics"] is True


def test_list_runs_shows_older_runs_too(root):
    store, song, first = _run_pipeline(root)
    second = run_song(
        "t",
        ScriptedClient(
            {
                "a": [],
                "b": scripted(draft_json("S1", {"P1": "四五六", "P2": "七八九"})),
            }
        ),
        root=root,
    )
    rows = list_runs(store, song)
    assert [r["run_id"] for r in rows] == [first.run_id, second.run_id]
    assert [r["current"] for r in rows] == [False, True]


def test_render_record_dispatches_by_agent(root):
    store, song, outcome = _run_pipeline(root)

    assert render_record(song, store, outcome.run_id, "a", None).startswith("A 的体会")
    assert "B 的译词" in render_record(song, store, outcome.run_id, "b", "S1")
    assert "C 的检查" in render_record(song, store, outcome.run_id, "c", "S1")
    assert "还没有 B 产物" in render_record(song, store, "run-nope", "b", "S1")
