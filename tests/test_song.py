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


# -- pattern -------------------------------------------------------------


def test_pattern_parses_and_sums_to_budget():
    raw = song_json(
        "x",
        ["S1"],
        [
            {
                "id": "P1",
                "section": "S1",
                "ja": "あ",
                "budget": {"min": 12, "max": 12},
                "pattern": [3, 4, 5],
            }
        ],
    )
    song = Song.from_json(raw)
    p = song.phrase("P1")
    assert p.pattern == (3, 4, 5)
    assert sum(p.parts) == 12
    assert p.to_json()["pattern"] == [3, 4, 5]


def test_pattern_defaults_to_one_part():
    raw = song_json("x", ["S1"], [phrase("P1", "S1", "あ", 7)])
    p = Song.from_json(raw).phrase("P1")
    assert p.pattern is None
    assert p.parts == (7,)
    assert "pattern" not in p.to_json()


def test_pattern_must_fit_budget():
    raw = song_json(
        "x",
        ["S1"],
        [
            {
                "id": "P1",
                "section": "S1",
                "ja": "あ",
                "budget": {"min": 12, "max": 12},
                "pattern": [3, 4, 4],
            }
        ],
    )
    with pytest.raises(SchemaError, match="pattern 合计 11"):
        Song.from_json(raw)


def test_pattern_allows_range_budget():
    raw = song_json(
        "x",
        ["S1"],
        [
            {
                "id": "P1",
                "section": "S1",
                "ja": "あ",
                "budget": {"min": 10, "max": 12},
                "pattern": [5, 6],
            }
        ],
    )
    assert Song.from_json(raw).phrase("P1").pattern == (5, 6)


@pytest.mark.parametrize("bad", [[], [3, 0], [3, -1], "3,4", [3, 4.5], [True, 2]])
def test_pattern_rejects_bad_shapes(bad):
    raw = song_json(
        "x",
        ["S1"],
        [
            {
                "id": "P1",
                "section": "S1",
                "ja": "あ",
                "budget": {"min": 12, "max": 12},
                "pattern": bad,
            }
        ],
    )
    with pytest.raises(SchemaError, match="pattern"):
        Song.from_json(raw)


def test_pattern_requires_budget():
    raw = song_json(
        "x", ["S1"], [{"id": "P1", "section": "S1", "ja": "あ", "pattern": [3]}]
    )
    with pytest.raises(SchemaError, match="就必须写 budget"):
        Song.from_json(raw)


def test_pattern_conflicts_with_keep_source():
    raw = song_json(
        "x",
        ["S1"],
        [
            {
                "id": "P1",
                "section": "S1",
                "ja": "あ",
                "keep_source": True,
                "pattern": [3],
            }
        ],
    )
    with pytest.raises(SchemaError, match="keep_source"):
        Song.from_json(raw)
