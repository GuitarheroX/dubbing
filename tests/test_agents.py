import pytest

from dubbing.agents import (
    build_lookup_request,
    build_mimic_request,
    build_writer_request,
    feedback_block,
    lookup_payload,
    mimic_payload,
    parse_draft_output,
    parse_perceiver_output,
    writer_payload,
)
from dubbing.models import LookupResult, PoetryHit, Violation

from helpers import analysis_json, make_analysis, make_draft, make_song, phrase

SONG = make_song(
    "t",
    ["V1", "C1"],
    [
        phrase("V1-01", "V1", "ああああ", 4),
        phrase("V1-02", "V1", "いいいい", 4),
        phrase("C1-01", "C1", "ううう", 3),
        phrase("C1-02", "C1", "ららら", keep_source=True),
    ],
)
ANALYSIS = make_analysis(["V1", "C1"])

HIT = PoetryHit(
    chunk_id="顾城_gucheng/一代人#s1",
    text="黑夜给了我黑色的眼睛\n我却用它寻找光明",
    score=0.7123,
    title="一代人",
    poet="顾城",
    poem_id="顾城_gucheng/一代人",
    stanza_idx=1,
)


def _refs(*hits: PoetryHit, section: str = "C1") -> LookupResult:
    return LookupResult(
        section=section,
        query_text="……",
        k=4,
        hits=hits,
        collection="poetry_stanza256_v1",
        index_id="poetry_stanza256_v1",
    )


# -- writer 的入参 --------------------------------------------------------


def test_writer_payload_separates_zh_from_prev_zh():
    data = writer_payload(
        SONG,
        ANALYSIS,
        focus="C1",
        accepted={"V1-01": "甲乙丙丁"},
        prev_texts={"C1-01": "戊己庚"},
    )
    rows = {r["id"]: r for r in data["phrases"]}

    assert rows["V1-01"]["zh"] == "甲乙丙丁"  # 定稿，只读参考
    assert "prev_zh" not in rows["V1-01"]
    assert "zh" not in rows["V1-02"]  # 还没译
    assert rows["C1-01"]["prev_zh"] == "戊己庚"  # 本段上一版，可改
    assert "zh" not in rows["C1-01"]


def test_writer_payload_carries_keep_source_and_no_budget():
    data = writer_payload(SONG, ANALYSIS, focus="C1")
    kept = next(r for r in data["phrases"] if r["id"] == "C1-02")
    assert kept["keep_source"] is True
    assert kept["budget"] is None


def test_writer_payload_has_whole_song_in_order():
    data = writer_payload(SONG, ANALYSIS, focus="C1")
    assert [r["id"] for r in data["phrases"]] == ["V1-01", "V1-02", "C1-01", "C1-02"]
    assert data["focus"] == "C1"
    assert data["analysis"]["overall"]["theme"]
    assert [s["id"] for s in data["sections"]] == ["V1", "C1"]


def test_writer_payload_rejects_unknown_focus():
    with pytest.raises(KeyError):
        writer_payload(SONG, ANALYSIS, focus="S9")


def test_writer_payload_has_no_instructions():
    """数据里不该出现指令性字段。"""
    data = writer_payload(SONG, ANALYSIS, focus="C1")
    raw = str(data)
    for word in ("do_not", "avoid", "forbidden", "写法建议"):
        assert word not in raw


# -- mimic 的入参（检索 + 改写） -----------------------------------------


def test_mimic_payload_is_writer_payload_plus_draft_and_references():
    draft = make_draft("C1", {"C1-01": "初版"})
    data = mimic_payload(
        SONG,
        ANALYSIS,
        focus="C1",
        draft=draft,
        references=_refs(HIT),
        accepted={"V1-01": "甲乙丙丁"},
        prev_texts={"C1-01": "上一版"},
    )

    assert data["focus"] == "C1"
    assert data["draft"]["phrases"] == [{"id": "C1-01", "zh": "初版"}]
    assert data["references"][0]["chunk_id"] == HIT.chunk_id
    assert data["references"][0]["text"].startswith("黑夜给了我")
    assert data["references"][0]["score"] == 0.7123

    rows = {r["id"]: r for r in data["phrases"]}
    assert rows["V1-01"]["zh"] == "甲乙丙丁"  # 其他段已定稿
    assert rows["C1-01"]["prev_zh"] == "上一版"  # 本段上一轮
    assert rows["V1-02"]["budget"] == {"min": 4, "max": 4}


def test_mimic_sees_the_whole_song_and_the_focus_section():
    """整首歌的日语原文都要在，否则参考诗句会把译词带偏。"""
    data = mimic_payload(SONG, ANALYSIS, focus="C1", draft=make_draft("C1", {"C1-01": "初版"}))

    assert [r["id"] for r in data["phrases"]] == ["V1-01", "V1-02", "C1-01", "C1-02"]
    assert {r["section"] for r in data["phrases"]} == {"V1", "C1"}
    assert data["phrases"][0]["ja"] == "ああああ"
    assert data["focus"] == "C1"
    assert data["references"] == []  # 没有检索结果也不报错


def test_mimic_payload_rejects_unknown_focus():
    with pytest.raises(KeyError):
        mimic_payload(SONG, ANALYSIS, focus="S9", draft=make_draft("C1", {}))


def test_mimic_request_keeps_feedback_out_of_data():
    request = build_mimic_request(
        SONG,
        ANALYSIS,
        focus="C1",
        draft=make_draft("C1", {"C1-01": "词"}),
        feedback=[Violation("C1-01", "OVER_BUDGET", "hard", expected=(3, 3), actual=5, fix="砍两个字")],
    )
    assert "砍两个字" in request.user
    assert "砍两个字" not in str(request.data)
    assert request.data["references"] == []


# -- mimic 的检索入参 ----------------------------------------------------


def test_lookup_query_is_only_the_focus_section_translation():
    draft = make_draft(
        "C1",
        {"C1-01": "把声音写成有重量", "C1-02": "保留原文不该出现", "V1-01": "别的段"},
    )
    data = lookup_payload(SONG, draft, focus="C1", k=4, collection="poetry_stanza256_v1")

    assert data["query_text"] == "把声音写成有重量"  # keep_source 与别的段都不参与
    assert data["k"] == 4
    assert data["collection"] == "poetry_stanza256_v1"


def test_lookup_query_is_empty_when_nothing_is_translated():
    data = lookup_payload(SONG, make_draft("C1", {}), focus="C1", k=4)
    assert data["query_text"] == ""


def test_lookup_request_carries_no_prompt():
    request = build_lookup_request(SONG, make_draft("C1", {"C1-01": "词"}), focus="C1", k=4)
    assert "不调用模型" in request.system
    assert request.user == ""


# -- 提示词的职责搬家 ----------------------------------------------------


def test_writer_prompt_drops_the_hard_budget_rule():
    system = build_writer_request(SONG, ANALYSIS, focus="C1").system
    assert "大概要唱几个字" in system
    assert "不用精确数字数" in system
    assert "必须落在它自己的 `budget` 区间里" not in system


def test_mimic_prompt_owns_the_hard_budget_rule_and_the_references():
    request = build_mimic_request(
        SONG, ANALYSIS, focus="C1", draft=make_draft("C1", {"C1-01": "词"})
    )
    assert "必须落在它自己的 `budget` 区间里" in request.system
    assert "不得整行照抄" in request.system
    assert "字数永远优先于模仿" in request.system


# -- writer / mimic 共用的产物解析 --------------------------------------


def test_parse_drops_unknown_kept_and_duplicate():
    obj = {
        "section": "C1",
        "phrases": [
            {"id": "C1-01", "zh": "甲乙丙"},
            {"id": "C1-99", "zh": "x"},
            {"id": "C1-02", "zh": "保留"},
            {"id": "C1-01", "zh": "重复"},
        ],
    }
    draft, warnings = parse_draft_output(obj, SONG, "C1")

    assert [p.id for p in draft.phrases] == ["C1-01"]
    assert draft.phrases[0].zh == "甲乙丙"
    assert len(warnings) == 3
    assert any("未知乐句" in w for w in warnings)
    assert any("保留原文" in w for w in warnings)
    assert any("重复出现" in w for w in warnings)


def test_parse_fills_missing_with_empty():
    draft, warnings = parse_draft_output({"section": "C1", "phrases": []}, SONG, "C1")
    assert [p.id for p in draft.phrases] == ["C1-01"]
    assert draft.phrases[0].zh == ""
    assert any("缺少译词" in w for w in warnings)


def test_parse_trusts_requested_section():
    obj = {"section": "V1", "phrases": [{"id": "C1-01", "zh": "甲乙丙"}]}
    draft, warnings = parse_draft_output(obj, SONG, "C1")
    assert draft.section == "C1"
    assert any("不符" in w for w in warnings)


def test_parse_drops_other_sections():
    obj = {"section": "C1", "phrases": [{"id": "C1-01", "zh": "甲乙丙"}, {"id": "V1-01", "zh": "丁戊己庚"}]}
    draft, warnings = parse_draft_output(obj, SONG, "C1")
    assert [p.id for p in draft.phrases] == ["C1-01"]
    assert any("其他 section" in w for w in warnings)


def test_parse_accepts_writer_output_without_chars():
    """writer 不再报字数，解析器不该因此报错。"""
    draft, warnings = parse_draft_output(
        {"section": "C1", "phrases": [{"id": "C1-01", "zh": "甲乙丙"}]}, SONG, "C1"
    )
    assert draft.phrases[0].chars is None
    assert warnings == []


# -- perceiver 的产物解析 ------------------------------------------------


def test_perceiver_parse_drops_unknown_sections():
    raw = analysis_json(["V1", "C1"])
    raw["sections"]["S9"] = "不存在"
    analysis, warnings = parse_perceiver_output(raw, SONG)
    assert "S9" not in analysis.sections
    assert any("不存在" in w for w in warnings)


def test_perceiver_parse_warns_on_missing_sections():
    raw = analysis_json(["V1"])
    analysis, warnings = parse_perceiver_output(raw, SONG)
    assert any("没有覆盖" in w for w in warnings)


# -- 返修反馈 ------------------------------------------------------------


def test_feedback_block_marks_hard_and_soft():
    block = feedback_block(
        [
            Violation("C1-02", "OVER_BUDGET", "hard", expected=(9, 9), actual=12, fix="超出 3 字，压到 9 字以内"),
            Violation("C1-04", "PUNCT", "soft", fix="去掉标点"),
        ]
    )
    assert "[必修] C1-02：超出 3 字，压到 9 字以内" in block
    assert "[建议] C1-04：去掉标点" in block


def test_feedback_block_empty_for_first_round():
    assert feedback_block([]) == ""


def test_feedback_never_leaks_into_data():
    """反馈只进消息，data 里不能出现。"""
    request = build_writer_request(
        SONG, ANALYSIS, focus="C1", feedback=[Violation("C1-01", "EMPTY", "hard", fix="补一句译词")]
    )
    assert "补一句译词" in request.user
    assert "补一句译词" not in str(request.data)


# -- pattern 透传 --------------------------------------------------------


def test_pattern_reaches_writer_only_when_annotated():
    song = make_song(
        "t",
        ["S1"],
        [
            dict(phrase("P1", "S1", "ああああ", 12), pattern=[3, 4, 5]),
            phrase("P2", "S1", "いいいい", 5),
        ],
    )
    data = writer_payload(song, make_analysis(["S1"]), focus="S1")
    rows = {r["id"]: r for r in data["phrases"]}

    assert rows["P1"]["pattern"] == [3, 4, 5]
    assert "pattern" not in rows["P2"]  # 没标注就不出现，默认整句一段


def test_pattern_does_not_change_reviewer_verdict():
    """reviewer 不检查 pattern：字数和 pattern 对不上也照样 pass。"""
    from dubbing.checks import review

    song = make_song(
        "t", ["S1"], [dict(phrase("P1", "S1", "あ", 6), pattern=[2, 2, 2])]
    )
    draft = make_draft("S1", {"P1": "一二三四五六"})
    assert review(song, "S1", draft, draft_run="r1").verdict == "pass"
