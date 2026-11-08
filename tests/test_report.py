"""人看的渲染：每个 agent 的产物长什么样，以及 runs / show 的数据来源。"""

from dubbing.llm import ScriptedClient
from dubbing.models import Analysis, Draft, LookupResult, PoetryHit, Review
from dubbing.orchestrator import run_song, store_for
from dubbing.report import (
    list_runs,
    render_analysis,
    render_draft,
    render_lookup,
    render_review,
    render_record,
)
from dubbing.song import load_song
from dubbing.view import build_view

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
    text = render_draft("r2", SONG, draft, agent="mimic_write")
    assert "一二三" in text and "四五六" in text
    assert "↳ 三字整" in text
    assert "3-3" in text
    assert "mimic" in text


def test_render_lookup_shows_query_and_hits():
    result = LookupResult(
        section="S1",
        query_text="一二三\n四五六",
        k=4,
        hits=(
            PoetryHit(
                chunk_id="顾城_gucheng/一代人#s1",
                text="黑夜给了我黑色的眼睛",
                score=0.7123,
                title="一代人",
                poet="顾城",
            ),
        ),
        collection="poetry_stanza256_v1",
        index_id="poetry_stanza256_v1",
    )
    text = render_lookup("r2", result)
    assert "一二三" in text
    assert "0.7123" in text
    assert "《一代人》" in text and "顾城" in text
    assert "黑夜给了我黑色的眼睛" in text


def test_render_lookup_handles_empty_and_skipped():
    empty = render_lookup("r2", LookupResult(section="S1", query_text="q", k=4))
    assert "命中：无" in empty

    skipped = render_lookup(
        "r2", LookupResult(section="S1", query_text="", k=4, skipped="empty-query")
    )
    assert "empty-query" in skipped


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
            "perceiver": scripted(analysis_json(["S1"])),
            "writer": scripted(draft_json("S1", {"P1": "一二三", "P2": "四五六"})),
        }
    )
    outcome = run_song("t", client, root=root)
    return store_for(root, "t"), load_song("t", root=root), outcome


def test_list_runs_marks_the_current_one(root):
    store, song, outcome = _run_pipeline(root)
    rows = list_runs(store, song)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == outcome.run_id
    assert row["current"] is True
    assert row["model"] == "scripted"
    assert row["has_perceiver"] is True
    assert (row["passed"], row["total"]) == (1, 1)
    assert row["status"] == "completed"
    assert row["lyrics"] is True


def test_list_runs_shows_older_runs_too(root):
    store, song, first = _run_pipeline(root)
    second = run_song(
        "t",
        ScriptedClient(
            {
                "perceiver": [],
                "writer": scripted(draft_json("S1", {"P1": "四五六", "P2": "七八九"})),
            }
        ),
        root=root,
    )
    rows = list_runs(store, song)
    assert [r["run_id"] for r in rows] == [first.run_id, second.run_id]
    assert [r["current"] for r in rows] == [False, True]


def test_render_record_dispatches_by_agent(root):
    store, song, outcome = _run_pipeline(root)

    assert render_record(song, store, outcome.run_id, "perceiver", None).startswith("perceiver")
    assert "writer" in render_record(song, store, outcome.run_id, "writer", "S1")
    assert "reviewer 的检查" in render_record(song, store, outcome.run_id, "reviewer", "S1")
    assert "还没有 writer 产物" in render_record(song, store, "run-nope", "writer", "S1")


def test_render_record_can_show_the_rag_side(root):
    """rag 管线的三个 agent 都能单独查看。"""
    write_song(root, "t", ["S1"], [phrase("P1", "S1", "あああ", 3)])
    client = ScriptedClient(
        {
            "perceiver": scripted(analysis_json(["S1"])),
            "writer": scripted(draft_json("S1", {"P1": "初版"})),
            "mimic": scripted(draft_json("S1", {"P1": "一二三"})),
        }
    )

    class FakeRetriever:
        k = 4
        collection = "c"
        index_id = "idx"

        def search(self, query_text, *, k=None, exclude=None):
            return [
                PoetryHit(
                    chunk_id="x#s1", text="黑夜给了我黑色的眼睛", score=0.5, title="一代人", poet="顾城"
                )
            ]

    outcome = run_song(
        "t", client, root=root, pipeline="rag", retriever=FakeRetriever(), only={"S1"}
    )
    store = store_for(root, "t", "rag")
    song = load_song("t", root=root)

    assert "初版" in render_record(song, store, outcome.run_id, "writer", "S1")
    assert "黑夜给了我黑色的眼睛" in render_record(song, store, outcome.run_id, "mimic_lookup", "S1")
    assert "一二三" in render_record(song, store, outcome.run_id, "mimic_write", "S1")
    # 装配视图时取的是 mimic_write（rag 管线的最终文本）
    assert build_view(song, store, outcome.run_id, pipeline="rag").row("P1").sing_text == "一二三"
