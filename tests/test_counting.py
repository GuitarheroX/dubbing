from dubbing.counting import contains_kana, count_syllables, has_punct


def test_counts_hanzi_only():
    assert count_syllables("像要沉下去又像要融化") == 10


def test_ignores_spaces_and_punct():
    assert count_syllables("像要沉下去 又像要融化") == 10
    assert count_syllables("一二三，四五六。") == 6
    assert count_syllables("，。！？ 　") == 0
    assert count_syllables("") == 0


def test_kana_counts_one_each():
    assert count_syllables("ベイベー") == 4


def test_latin_by_vowel_groups():
    assert count_syllables("baby") == 2
    assert count_syllables("stray") == 1
    assert count_syllables("rhythm") == 1
    assert count_syllables("hi") == 1


def test_digits_one_each():
    assert count_syllables("1999") == 4


def test_mixed_line():
    assert count_syllables("baby 不要走") == 2 + 3


def test_kana_detection():
    assert contains_kana("アイウ")
    assert contains_kana("中あ")
    assert not contains_kana("全部中文")
    assert not contains_kana("中文 baby")


def test_punct_detection():
    assert has_punct("一二三，四")
    assert has_punct("一二三 四") is False
    assert not has_punct("一二三四")
