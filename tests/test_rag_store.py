"""本地 Chroma：写入、查询、排除、manifest。

没装 chromadb 就整体跳过（rag 只依赖可选包，基础安装不该被动到）。
"""

import pytest

from dubbing.models import PoetryHit
from dubbing.rag.store import ChromaStore, StoreError, read_manifest, write_manifest

pytest.importorskip("chromadb")


def _store(tmp_path, name="poetry_stanza256_v1"):
    return ChromaStore(tmp_path / "chroma", name, create=True)


def _seed(store):
    store.upsert(
        ids=["a#s1", "b#s1", "c#s1"],
        documents=["雨夜", "末班车", "稻草人"],
        metadatas=[
            {"title": "雨", "poet": "甲", "poem_id": "甲_jia/雨", "stanza_idx": 1},
            {"title": "末班车", "poet": "乙", "poem_id": "乙_yi/末班车", "stanza_idx": 1},
            {"title": "稻草人", "poet": "丙", "poem_id": "丙_bing/稻草人", "stanza_idx": 1},
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )


def test_upsert_and_query_roundtrip(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    assert store.count() == 3

    hits = store.query([1.0, 0.0, 0.0], 2)
    assert isinstance(hits[0], PoetryHit)
    assert hits[0].chunk_id == "a#s1"
    assert hits[0].text == "雨夜"
    assert hits[0].title == "雨"
    assert hits[0].poet == "甲"
    assert hits[0].poem_id == "甲_jia/雨"
    assert hits[0].stanza_idx == 1
    assert hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert len(hits) == 2


def test_upsert_is_idempotent(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    _seed(store)  # 同一批 id 再来一次：覆盖而不是报错
    assert store.count() == 3


def test_query_honours_exclude(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    hits = store.query([1.0, 0.0, 0.0], 2, exclude={"a#s1"})
    assert [h.chunk_id for h in hits] == ["b#s1", "c#s1"]


def test_query_with_zero_k_returns_nothing(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    assert store.query([1.0, 0.0, 0.0], 0) == []


def test_missing_collection_is_a_clear_error(tmp_path):
    with pytest.raises(StoreError, match="没有 collection"):
        ChromaStore(tmp_path / "chroma", "nope_not_there")


def test_reset_empties_the_collection(tmp_path):
    store = _store(tmp_path)
    _seed(store)
    store.reset()
    assert store.count() == 0


def test_manifest_round_trip_and_index_id(tmp_path):
    store = _store(tmp_path)
    assert store.index_id == "poetry_stanza256_v1"  # 没有 manifest 就退回 collection 名
    write_manifest(store.path, {"index_id": "poetry_stanza256_bge_v1", "chunks": 3})
    assert read_manifest(store.path)["chunks"] == 3
    assert store.index_id == "poetry_stanza256_bge_v1"


def test_upsert_rejects_mismatched_lengths(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(StoreError, match="长度不一致"):
        store.upsert(ids=["a"], documents=["x", "y"], metadatas=[{}], embeddings=[[1.0]])
