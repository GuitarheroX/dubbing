"""C 的规则检查。"""

from dubbing.checks import review
from dubbing.models import Changes, Review

from helpers import make_draft, make_song, phrase

SONG = make_song(
    "t",
    ["S1"],
    [
        phrase("P1", "S1", "朝", 5),
        phrase("P2", "S1", "夜", 5),
        phrase("P3", "S1", "ららら", keep_source=True),
    ],
)


def _codes(rev, pid):
    return {v.code for v in rev.violations if v.phrase_id == pid}


def test_over_and_under():
    draft = make_draft("S1", {"P1": "一二三四五六", "P2": "一二"})
    rev = review(SONG, "S1", draft, draft_run="r1")

    assert rev.verdict == "fail"
    assert _codes(rev, "P1") == {"OVER_BUDGET"}
    assert _codes(rev, "P2") == {"UNDER_BUDGET"}
    assert [v.code for v in rev.violations if v.phrase_id == "P1"] == ["OVER_BUDGET"]
    assert rev.hard[0].expected == (5, 5)
    assert rev.hard[0].actual == 6


def test_pass_when_every_phrase_fits():
    draft = make_draft("S1", {"P1": "一二三四五", "P2": "六七八九十"})
    rev = review(SONG, "S1", draft, draft_run="r1")
    assert rev.verdict == "pass"
    assert rev.violations == ()


def test_empty_is_hard():
    draft = make_draft("S1", {"P1": "一二三四五", "P2": "   "})
    rev = review(SONG, "S1", draft, draft_run="r1")
    assert _codes(rev, "P2") == {"EMPTY"}
    assert rev.verdict == "fail"


def test_kana_leftover():
    draft = make_draft("S1", {"P1": "アイウ", "P2": "一二三四五"})
    rev = review(SONG, "S1", draft, draft_run="r1")
    assert "NON_ZH" in _codes(rev, "P1")
    assert "UNDER_BUDGET" in _codes(rev, "P1")


def test_duplicate_only_when_sources_differ():
    same_text = "一二三四五"
    draft = make_draft("S1", {"P1": same_text, "P2": same_text})
    rev = review(SONG, "S1", draft, draft_run="r1")
    assert _codes(rev, "P1") == {"DUPLICATE"}
    assert _codes(rev, "P2") == {"DUPLICATE"}

    # 原文相同（副歌重复）时，译词相同是正确的
    song = make_song(
        "t", ["S1"], [phrase("P1", "S1", "朝", 5), phrase("P2", "S1", "朝", 5)]
    )
    rev = review(song, "S1", draft, draft_run="r1")
    assert rev.violations == ()
    assert rev.verdict == "pass"


def test_soft_violations_do_not_fail_the_round():
    draft = make_draft(
        "S1", {"P1": "一二三四五，", "P2": "六七八九十"}, chars={"P1": 4}
    )
    rev = review(SONG, "S1", draft, draft_run="r1")
    assert _codes(rev, "P1") == {"PUNCT", "CHAR_MISMATCH"}
    assert all(v.severity == "soft" for v in rev.violations)
    assert rev.verdict == "pass"


def test_keep_source_is_never_checked():
    draft = make_draft("S1", {"P1": "一二三四五", "P2": "六七八九十", "P3": "ららら"})
    rev = review(SONG, "S1", draft, draft_run="r1")
    assert "P3" not in {v.phrase_id for v in rev.violations}


def test_changes_are_recorded():
    prev = make_draft("S1", {"P1": "一二三四五", "P2": "六七八九十"})
    cur = make_draft("S1", {"P1": "一二三四", "P2": "六七八九十"})
    rev = review(SONG, "S1", cur, draft_run="r2", prev_draft=prev)
    assert rev.changes.rewritten == ("P1",)
    assert rev.changes.unchanged == ("P2",)


def test_first_round_marks_everything_rewritten():
    cur = make_draft("S1", {"P1": "一二三四五", "P2": "六七八九十"})
    rev = review(SONG, "S1", cur, draft_run="r1")
    assert rev.changes.rewritten == ("P1", "P2")
    assert rev.changes.unchanged == ()


def test_regressions_flag_lines_broken_by_a_repair():
    prev = make_draft("S1", {"P1": "一二三四五", "P2": "六七八九十"})
    prev_review = Review(
        section="S1",
        draft_run="r1",
        verdict="pass",
        violations=(),
        changes=Changes(rewritten=("P1", "P2")),
    )
    cur = make_draft("S1", {"P1": "一二三四五", "P2": "六七八九十十"})  # P2 变 6 字
    rev = review(SONG, "S1", cur, draft_run="r2", prev_draft=prev, prev_review=prev_review)

    assert rev.regressions == ("P2",)
    assert rev.verdict == "fail"
    assert "P2" in rev.failing_ids()
