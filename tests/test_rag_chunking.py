"""诗节切块：一个诗节 = 一个 chunk，不设重叠，token 上限 256。

这是 RAG 资产里唯一"规则很多、又不依赖任何重依赖"的部分，所以测得细一点。
"""

from dubbing.rag.chunking import (
    DEFAULT_MAX_TOKENS,
    Chunk,
    Poem,
    count_tokens_by_chars,
    corpus_root,
    iter_chunks,
    iter_poem_paths,
    parse_poem,
    poem_fingerprint,
)

POEM = """title:稻草人
date:20030330

一群鸟在离开田野好奇地
望着你，你好像无边无涯
初冬的轮子和寒霜
问你是否一起回家

稻草人说，我可以
轻松一个漫长的冬季
风很大
吹落了稻草人的帽子和面孔

光溜的躯体仍然站立
稻草人有着自己的哲学
时间醒了
白色的雪真的是你最亲的人
"""


def _write_poem(root, poet_dir, name, text):
    d = root / poet_dir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.pt"
    path.write_text(text, encoding="utf-8")
    return path


# -- 解析 ----------------------------------------------------------------


def test_parse_poem_reads_header_and_stanzas(tmp_path):
    path = _write_poem(tmp_path, "00913_00913", "稻草人", POEM)
    poem = parse_poem(path)

    assert poem is not None
    assert poem.title == "稻草人"
    assert poem.date == "20030330"
    assert poem.poet == "00913"  # 目录名里最后一个下划线之前的部分
    assert poem.poem_id == "00913_00913/稻草人"
    assert len(poem.stanzas) == 3
    assert poem.stanzas[0].startswith("一群鸟在离开田野")
    assert poem.stanzas[2].endswith("最亲的人")
    # 诗节内部保留换行（现代诗的断行就是句式）
    assert "\n" in poem.stanzas[0]


def test_parse_poem_falls_back_to_filename_without_header(tmp_path):
    path = _write_poem(tmp_path, "某人_someone", "无题", "第一行\n第二行\n\n第三行\n第四行更多字")
    poem = parse_poem(path)
    assert poem is not None
    assert poem.title == "无题"
    assert poem.date == ""
    assert poem.stanzas == ("第一行\n第二行", "第三行\n第四行更多字")


def test_parse_poem_skips_empty_and_too_short(tmp_path):
    assert parse_poem(_write_poem(tmp_path, "p_x", "空", "title:空\n\n\n")) is None
    assert parse_poem(_write_poem(tmp_path, "p_x", "短", "title:短\n\n短句")) is None


def test_poem_fingerprint_ignores_whitespace(tmp_path):
    a = parse_poem(_write_poem(tmp_path, "p_x", "a", "title:a\n\n一二三四五六七八九十\n\n甲乙丙丁戊己庚辛壬癸"))
    b = parse_poem(_write_poem(tmp_path, "p_y", "b", "title:b\n\n一二三四五六七八九十\n\n\n甲乙丙丁戊己庚辛壬癸"))
    assert a is not None and b is not None
    assert poem_fingerprint(a) == poem_fingerprint(b)


# -- 语料定位 ------------------------------------------------------------


def test_corpus_root_accepts_repo_root_or_data_dir(tmp_path):
    data = tmp_path / "data" / "诗人_poet"
    data.mkdir(parents=True)
    (data / "某诗.pt").write_text("title:某诗\n\n一二三四五六七八九十", encoding="utf-8")

    assert corpus_root(tmp_path) == tmp_path / "data"
    assert corpus_root(tmp_path / "data") == tmp_path / "data"


def test_iter_poem_paths_walks_poet_directories(tmp_path):
    _write_poem(tmp_path, "b_second", "二", "title:二\n\n一二三四五六七八九十")
    _write_poem(tmp_path, "a_first", "一", "title:一\n\n一二三四五六七八九十")
    (tmp_path / "a_first" / "task.py").write_text("x", encoding="utf-8")

    names = [p.name for p in iter_poem_paths(tmp_path)]
    assert names == ["一.pt", "二.pt"]  # 按诗人目录排序


# -- 切块 ----------------------------------------------------------------


def test_one_chunk_per_stanza_and_no_overlap(tmp_path):
    poem = parse_poem(_write_poem(tmp_path, "p_x", "稻草人", POEM))
    assert poem is not None
    chunks = list(iter_chunks(poem))

    assert len(chunks) == 3
    assert [c.stanza_idx for c in chunks] == [1, 2, 3]
    assert all(c.n_parts == 1 and c.part_idx == 1 for c in chunks)
    assert not any(c.continued for c in chunks)
    assert chunks[0].text == poem.stanzas[0]
    assert "\n\n" not in "".join(c.text for c in chunks)  # 空行只当分隔符，不进正文


def test_chunk_metadata_carries_title_and_counts(tmp_path):
    poem = parse_poem(_write_poem(tmp_path, "顾城_gucheng", "一代人", "title:一代人\n\n黑夜给了我黑色的眼睛\n我却用它寻找光明"))
    assert poem is not None
    chunk = next(iter(iter_chunks(poem)))
    meta = chunk.metadata()

    assert meta["title"] == "一代人"  # 诗题必须留在 metadata 里
    assert meta["poet"] == "顾城"
    assert meta["poem_id"] == "顾城_gucheng/一代人"
    assert meta["stanza_idx"] == 1
    assert meta["chars"] == 18
    assert meta["tokens"] == 18
    assert chunk.chunk_id == "顾城_gucheng/一代人#s1"
    # metadata 只放标量：值里不能出现 list/dict
    assert all(isinstance(v, (str, int, bool)) for v in meta.values())


def test_chunk_ids_are_unique_and_stable(tmp_path):
    poem = parse_poem(_write_poem(tmp_path, "p_x", "稻草人", POEM))
    assert poem is not None
    first = [c.chunk_id for c in iter_chunks(poem)]
    second = [c.chunk_id for c in iter_chunks(poem)]
    assert first == second
    assert len(set(first)) == len(first)


def test_long_stanza_is_split_at_line_boundaries(tmp_path):
    poem = Poem(
        poem_id="p_x/长",
        poet="p",
        title="长",
        date="",
        stanzas=("一二三四五\n六七八九十\n甲乙丙丁戊",),
    )
    chunks = list(iter_chunks(poem, count_tokens=count_tokens_by_chars, max_tokens=10))

    assert len(chunks) == 2
    assert chunks[0].text == "一二三四五\n六七八九十"
    assert chunks[1].text == "甲乙丙丁戊"
    assert [c.n_parts for c in chunks] == [2, 2]
    assert [c.part_idx for c in chunks] == [1, 2]
    assert [c.chunk_id for c in chunks] == ["p_x/长#s1p1", "p_x/长#s1p2"]
    assert chunks[1].continued is True
    # 拆出来的块共享同一份 metadata
    assert chunks[0].metadata()["stanza_idx"] == chunks[1].metadata()["stanza_idx"] == 1


def test_single_line_longer_than_cap_is_split_by_punctuation(tmp_path):
    poem = Poem(
        poem_id="p_x/散文诗",
        poet="p",
        title="散文诗",
        date="",
        stanzas=("第一句，第二句，第三句，第四句。",),
    )
    chunks = list(iter_chunks(poem, count_tokens=count_tokens_by_chars, max_tokens=8))

    assert len(chunks) > 1
    assert all(c.tokens <= 8 for c in chunks)
    # 不丢字：所有片段拼回去就是原文
    assert "".join(c.text for c in chunks) == "第一句，第二句，第三句，第四句。"


def test_absurdly_long_line_is_hard_split(tmp_path):
    poem = Poem(poem_id="p_x/病态", poet="p", title="病态", date="", stanzas=("甲" * 500,))
    chunks = list(iter_chunks(poem, count_tokens=count_tokens_by_chars, max_tokens=64))

    assert all(c.tokens <= 64 for c in chunks)
    assert "".join(c.text for c in chunks) == "甲" * 500


def test_default_cap_is_256_tokens():
    assert DEFAULT_MAX_TOKENS == 256


def test_count_tokens_by_chars_ignores_whitespace():
    assert count_tokens_by_chars("一二 三\n四") == 4
    assert count_tokens_by_chars("") == 0


def test_chunk_to_json_keeps_text_for_the_intermediate_file(tmp_path):
    poem = parse_poem(_write_poem(tmp_path, "p_x", "稻草人", POEM))
    assert poem is not None
    chunk = next(iter(iter_chunks(poem)))
    payload = chunk.to_json()
    assert isinstance(chunk, Chunk)
    assert payload["text"] == chunk.text
    assert payload["chunk_id"] == chunk.chunk_id
