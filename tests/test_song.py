import pytest

from dubbing.models import SchemaError, Song
from dubbing.song import list_songs, load_song

from helpers import phrase, song_json


def test_loads_kotozute(data_root):
    song = load_song("kotozute", root=data_root)
    assert song.title == "言伝"
    assert song.section_ids == ("V1", "V2", "B1", "C1")
    assert len(song.phrases) == 21
    assert [len(song.phrases_of(s)) for s in song.section_ids] == [3, 3, 4, 11]
    assert sum(p.budget.max for p in song.phrases) == 150
    assert all(p.budget.min == p.budget.max for p in song.phrases)
    assert song.phrases[-1].ja == "飛ぶ"
    assert song.phrases[-1].budget.max == 2


def test_lists_songs(data_root):
    assert "kotozute" in list_songs(root=data_root)


def test_budget_must_be_ordered():
    raw = song_json("x", ["S1"], [{"id": "P1", "section": "S1", "ja": "あ", "budget": {"min": 9, "max": 3}}])
    with pytest.raises(SchemaError, match="区间颠倒"):
        Song.from_json(raw)


def test_keep_source_must_not_carry_budget():
    raw = song_json(
        "x",
        ["S1"],
        [{"id": "P1", "section": "S1", "ja": "あ", "keep_source": True, "budget": {"min": 3, "max": 3}}],
    )
    with pytest.raises(SchemaError, match="keep_source"):
        Song.from_json(raw)


def test_phrase_ids_unique():
    raw = song_json(
        "x",
        ["S1"],
        [phrase("P1", "S1", "あ", 3), phrase("P1", "S1", "い", 3)],
    )
    with pytest.raises(SchemaError, match="phrase id 重复"):
        Song.from_json(raw)


def test_phrase_must_belong_to_known_section():
    raw = song_json("x", ["S1"], [phrase("P1", "S9", "あ", 3)])
    with pytest.raises(SchemaError, match="未知 section"):
        Song.from_json(raw)


def test_song_needs_a_translatable_phrase():
    raw = song_json("x", ["S1"], [phrase("P1", "S1", "あ", keep_source=True)])
    with pytest.raises(SchemaError, match="没有任何可译乐句"):
        Song.from_json(raw)
