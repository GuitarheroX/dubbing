"""baseline 的输入拼装。"""

from baseline.run_baseline import compose, lyrics_block, suspicious_lines

from helpers import make_song, phrase


def _song():
    return make_song(
        "t",
        ["S1", "S2"],
        [
            phrase("A-01", "S1", "ああああ", 4),
            dict(phrase("A-02", "S1", "いいいいいい", 12), pattern=[3, 4, 5]),
            phrase("B-01", "S2", "らら", keep_source=True),
            phrase("B-02", "S2", "ううう", 3),
        ],
    )


def test_lyrics_block_layout():
    block, skipped = lyrics_block(_song())
    lines = block.splitlines()

    assert lines[0] == "测试歌"  # 第一行是标题
    assert lines[1] == "ああああ 4"  # 没 pattern 就只有总数
    assert lines[2] == "いいいいいい 12 3 4 5"  # 有 pattern 再接分段
    assert skipped == ["B-01"]  # 保留原文的跳过
    assert lines[-1] == "ううう 3"


def test_keep_source_lines_are_not_emitted():
    block, _ = lyrics_block(_song())
    assert "らら" not in block


def test_compose_puts_prompt_in_system_and_lyrics_in_user():
    system, user = compose(_song())
    assert system.rstrip().endswith("歌词：")
    assert user.startswith("测试歌")
    assert "ああああ 4" in user


def test_suspicious_lines_flags_numbers():
    assert suspicious_lines("夜色之中街灯璀璨华丽\n下一句") == []
    assert suspicious_lines("1. 夜色\n2. 下一句")
    assert suspicious_lines("夜色之中（10）\n下一句")


def test_prompt_says_not_to_output_the_title():
    from baseline.run_baseline import PROMPT_PATH

    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "第一行是标题" in text
    assert "不要翻译它" in text


def test_prompt_asks_for_lyrics_only():
    from baseline.run_baseline import PROMPT_PATH

    text = PROMPT_PATH.read_text(encoding="utf-8")
    assert "只输出中文歌词" in text
    assert "不要输出序号" in text
    assert text.rstrip().endswith("歌词：")
