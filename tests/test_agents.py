import pytest

from dubbing.agents import (
    b_payload,
    build_b_request,
    feedback_block,
    parse_a_output,
    parse_b_output,
)

from helpers import analysis_json, make_analysis, make_song, phrase

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


# -- B 的入参 ------------------------------------------------------------


def test_b_payload_separates_zh_from_prev_zh():
    data = b_payload(
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


def test_b_payload_carries_keep_source_and_no_budget():
    data = b_payload(SONG, ANALYSIS, focus="C1")
    kept = next(r for r in data["phrases"] if r["id"] == "C1-02")
    assert kept["keep_source"] is True
    assert kept["budget"] is None


def test_b_payload_has_whole_song_in_order():
    data = b_payload(SONG, ANALYSIS, focus="C1")
    assert [r["id"] for r in data["phrases"]] == ["V1-01", "V1-02", "C1-01", "C1-02"]
    assert data["focus"] == "C1"
    assert data["analysis"]["overall"]["theme"]
    assert [s["id"] for s in data["sections"]] == ["V1", "C1"]


def test_b_payload_rejects_unknown_focus():
    with pytest.raises(KeyError):
        b_payload(SONG, ANALYSIS, focus="S9")


def test_b_payload_has_no_instructions():
    """数据里不该出现指令性字段。"""
    data = b_payload(SONG, ANALYSIS, focus="C1")
    raw = str(data)
    for word in ("do_not", "avoid", "forbidden", "写法建议"):
        assert word not in raw


# -- B 的产物解析 --------------------------------------------------------


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
    draft, warnings = parse_b_output(obj, SONG, "C1")

    assert [p.id for p in draft.phrases] == ["C1-01"]
    assert draft.phrases[0].zh == "甲乙丙"
    assert len(warnings) == 3
    assert any("未知乐句" in w for w in warnings)
    assert any("保留原文" in w for w in warnings)
    assert any("重复出现" in w for w in warnings)


def test_parse_fills_missing_with_empty():
    draft, warnings = parse_b_output({"section": "C1", "phrases": []}, SONG, "C1")
    assert [p.id for p in draft.phrases] == ["C1-01"]
    assert draft.phrases[0].zh == ""
    assert any("缺少译词" in w for w in warnings)


def test_parse_trusts_requested_section():
    obj = {"section": "V1", "phrases": [{"id": "C1-01", "zh": "甲乙丙"}]}
    draft, warnings = parse_b_output(obj, SONG, "C1")
    assert draft.section == "C1"
    assert any("不符" in w for w in warnings)


def test_parse_drops_other_sections():
    obj = {"section": "C1", "phrases": [{"id": "C1-01", "zh": "甲乙丙"}, {"id": "V1-01", "zh": "丁戊己庚"}]}
    draft, warnings = parse_b_output(obj, SONG, "C1")
    assert [p.id for p in draft.phrases] == ["C1-01"]
    assert any("其他 section" in w for w in warnings)


# -- A 的产物解析 --------------------------------------------------------


def test_a_parse_drops_unknown_sections():
    raw = analysis_json(["V1", "C1"])
    raw["sections"]["S9"] = "不存在"
    analysis, warnings = parse_a_output(raw, SONG)
    assert "S9" not in analysis.sections
    assert any("不存在" in w for w in warnings)


def test_a_parse_warns_on_missing_sections():
    raw = analysis_json(["V1"])
    analysis, warnings = parse_a_output(raw, SONG)
    assert any("没有覆盖" in w for w in warnings)


# -- 返修反馈 ------------------------------------------------------------


def test_feedback_block_marks_hard_and_soft():
    from dubbing.models import Violation

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
    from dubbing.models import Violation

    request = build_b_request(
        SONG, ANALYSIS, focus="C1", feedback=[Violation("C1-01", "EMPTY", "hard", fix="补一句译词")]
    )
    assert "补一句译词" in request.user
    assert "补一句译词" not in str(request.data)
