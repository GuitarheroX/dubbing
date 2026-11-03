import yaml
import pytest

from dubbing.import_lyrics import (
    Block,
    LyricsImportError,
    PhraseSpec,
    build_song,
    dump_song,
    infer_roles,
    parse_lyrics,
)
from dubbing.models import Song


SAMPLE = """\
## 测试歌 [test-song]
あ 3
い 5 2 3

う 4
え 4
お 4

## 第二首 [song-two]
か *这句不用翻译*
き 4
"""


def test_parses_songs_blocks_and_budgets():
    songs = parse_lyrics(SAMPLE)
    assert [s.song_id for s in songs] == ["test-song", "song-two"]
    assert [s.title for s in songs] == ["测试歌", "第二首"]

    first = songs[0]
    assert [len(b) for b in first.blocks] == [2, 3]
    assert first.blocks[0].phrases[0].budget == 3
    assert first.blocks[0].phrases[0].pattern is None
    assert first.blocks[0].phrases[1].budget == 5
    assert first.blocks[0].phrases[1].pattern == (2, 3)


def test_keep_source_phrase_drops_budget():
    songs = parse_lyrics(SAMPLE)
    p = songs[1].blocks[0].phrases[0]
    assert p.ja == "か"
    assert p.keep_source is True
    assert p.budget is None
    assert p.notes == "这句不用翻译"


def test_short_pattern_is_completed_and_long_pattern_is_dropped():
    songs = parse_lyrics(
        """\
## x [x]
あ 5 2
い 4 3 3
"""
    )
    a, b = songs[0].blocks[0].phrases
    assert a.budget == 5
    assert a.pattern == (2, 3)
    assert "已补成" in a.notes
    assert b.budget == 4
    assert b.pattern is None
    assert "已忽略分句" in b.notes
    assert len(songs[0].warnings) == 2


def test_strict_rejects_mismatched_pattern():
    with pytest.raises(LyricsImportError, match="超过总字数"):
        parse_lyrics("## x [x]\nあ 4 3 3\n", strict=True)
    with pytest.raises(LyricsImportError, match="已补成"):
        parse_lyrics("## x [x]\nあ 5 2\n", strict=True)


def test_build_song_has_valid_section_and_phrase_ids():
    song = build_song(parse_lyrics(SAMPLE)[0])
    assert song.section_ids == ("V1", "C1")
    assert song.phrase_ids == ("V1-01", "V1-02", "C1-01", "C1-02", "C1-03")
    assert song.phrase("V1-02").pattern == (2, 3)


def test_dump_song_round_trips_through_yaml():
    song = build_song(parse_lyrics(SAMPLE)[0])
    raw = yaml.safe_load(dump_song(song))
    again = Song.from_json(raw)
    assert again.to_json() == song.to_json()


def test_infer_roles_marks_repeated_blocks_as_chorus():
    blocks = [
        Block([PhraseSpec("a", 1), PhraseSpec("b", 1), PhraseSpec("c", 1)]),
        Block([PhraseSpec("d", 1), PhraseSpec("e", 1), PhraseSpec("f", 1)]),
        Block([PhraseSpec("a", 1), PhraseSpec("b", 1), PhraseSpec("c", 1)]),
    ]
    assert infer_roles(blocks) == ["chorus", "pre-chorus", "chorus"]
