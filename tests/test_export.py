"""导出歌词文件。"""

from dubbing.export import side_by_side, write_lyrics, zh_lyrics
from dubbing.models import Changes, Review, SectionState, State, ViewRow
from dubbing.models import View

from helpers import make_song, phrase


def _view(song, texts, *, keep=()):
    rows = []
    for p in song.phrases:
        if p.id in keep:
            rows.append(ViewRow(p.id, p.section, p.ja, p.budget, "ja", p.ja, "skip"))
        elif p.id in texts:
            rows.append(ViewRow(p.id, p.section, p.ja, p.budget, "zh", texts[p.id], "pass"))
        else:
            rows.append(ViewRow(p.id, p.section, p.ja, p.budget, None, "", "pending"))
    return View(song_id=song.id, rows=tuple(rows))


SONG = make_song(
    "t",
    ["S1", "S2"],
    [
        phrase("P1", "S1", "あああ", 3),
        phrase("P2", "S1", "いいい", keep_source=True),
        phrase("P3", "S2", "ううう", 3),
    ],
)


def test_zh_lyrics_has_sections_and_marks_untranslated():
    text = zh_lyrics(SONG, _view(SONG, {"P1": "一二三"}, keep={"P2"}))
    assert "一二三" in text
    assert "いいい" in text          # keep_source 直接取原文
    assert "〈未译〉ううう" in text   # 未译要显眼
    assert "[S1]" in text and "[S2]" in text


def test_side_by_side_counts_chars():
    md = side_by_side(SONG, _view(SONG, {"P1": "一二三"}, keep={"P2"}))
    assert "| あああ | 一二三 | 3 / 3 | pass |" in md
    assert "| いいい | いいい | — / - | skip |" in md
    assert "已定稿 2 句" in md


def test_write_lyrics_creates_two_files(tmp_path):
    paths = write_lyrics(SONG, _view(SONG, {"P1": "一二三"}), tmp_path)
    assert [p.name for p in paths] == ["lyrics.zh.txt", "lyrics.md"]
    assert all(p.is_file() for p in paths)
