"""向量文件（vectors.npy + vectors.json）：配对校验是这里的重点。"""

import json

import pytest

from dubbing.rag.vectors import (
    VectorFileError,
    VectorMeta,
    count_lines,
    file_sha1,
    load_vectors,
    save_vectors,
    sidecar_path,
)


def _chunks(tmp_path, n=3):
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"chunk_id": f"c#{i}", "text": f"诗句{i}"}, ensure_ascii=False) + "\n")
    return path


def _vectors(tmp_path, rows, chunks, *, count=None, dim=4, dtype="float16", sha=None):
    meta = VectorMeta(
        model="BAAI/bge-large-zh-v1.5",
        dim=dim,
        count=count if count is not None else len(rows),
        dtype=dtype,
        chunks_file=str(chunks),
        chunks_sha1=sha if sha is not None else file_sha1(chunks),
    )
    path = tmp_path / "vectors.npy"
    save_vectors(path, rows, meta)
    return path


def test_save_and_load_round_trip(tmp_path):
    chunks = _chunks(tmp_path, 3)
    path = _vectors(tmp_path, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]], chunks)

    vectors, meta = load_vectors(path, expected_count=3, chunks_file=chunks, chunks_sha1=file_sha1(chunks))

    assert vectors.shape == (3, 4)
    assert meta.model == "BAAI/bge-large-zh-v1.5"
    assert meta.count == 3
    assert meta.dim == 4
    assert meta.dtype == "float16"
    assert meta.chunks_sha1 == file_sha1(chunks)
    assert sidecar_path(path).name == "vectors.json"
    # dtype 用了 float16：值仍读得回来（近似）
    assert float(vectors[0][0]) == pytest.approx(1.0)


def test_mmap_mode_does_not_load_everything(tmp_path):
    chunks = _chunks(tmp_path, 2)
    path = _vectors(tmp_path, [[1.0, 0.0], [0.0, 1.0]], chunks, dim=2)
    vectors, _ = load_vectors(path, mmap=True)
    assert vectors.shape == (2, 2)  # 形状可读；这里只断言能正常打开


def test_missing_vector_file_is_reported(tmp_path):
    with pytest.raises(VectorFileError, match="找不到向量文件"):
        load_vectors(tmp_path / "nope.npy")


def test_missing_sidecar_is_reported(tmp_path):
    chunks = _chunks(tmp_path)
    path = _vectors(tmp_path, [[1.0, 0.0]], chunks, count=1, dim=2, dtype="float32")
    sidecar_path(path).unlink()
    with pytest.raises(VectorFileError, match="找不到向量文件的说明"):
        load_vectors(path)


def _write_raw_pair(tmp_path, rows, sidecar):
    """绕过 save_vectors 的写入期校验，手工造一份"自相矛盾"的向量文件。"""
    import numpy as np

    path = tmp_path / "vectors.npy"
    np.save(path, np.asarray(rows, dtype="float32"))
    sidecar_path(path).write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
    return path


def test_count_mismatch_with_chunks_is_reported(tmp_path):
    chunks = _chunks(tmp_path, 3)
    # sidecar 说自己有 3 条，实际只写了 2 条向量
    path = _write_raw_pair(
        tmp_path,
        [[1.0, 0.0], [0.0, 1.0]],
        {"model": "m", "dim": 2, "count": 3, "dtype": "float32"},
    )
    with pytest.raises(VectorFileError, match="3"):
        load_vectors(path, chunks_file=chunks)


def test_expected_count_mismatch_is_reported(tmp_path):
    chunks = _chunks(tmp_path, 3)
    path = _vectors(tmp_path, [[1.0, 0.0], [0.0, 1.0]], chunks, count=2, dim=2, dtype="float32")
    with pytest.raises(VectorFileError, match="与 chunks 文件的行数 3 不一致"):
        load_vectors(path, expected_count=3)


def test_dim_mismatch_is_reported(tmp_path):
    chunks = _chunks(tmp_path, 1)
    path = _write_raw_pair(
        tmp_path,
        [[1.0, 0.0, 0.0]],
        {"model": "m", "dim": 8, "count": 1, "dtype": "float32"},
    )
    with pytest.raises(VectorFileError, match="向量维度是 3，但 vectors.json 里写的是 8"):
        load_vectors(path)


def test_chunks_sha_mismatch_is_reported(tmp_path):
    chunks = _chunks(tmp_path, 1)
    path = _vectors(tmp_path, [[1.0, 0.0]], chunks, count=1, dim=2, dtype="float32")
    # 切块文件被改过（换了参数），向量还是旧的
    with chunks.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"chunk_id": "c#9", "text": "新加的一节"}, ensure_ascii=False) + "\n")

    with pytest.raises(VectorFileError, match="对不上"):
        load_vectors(path, chunks_file=chunks, chunks_sha1=file_sha1(chunks))


def test_sidecar_without_sha_skips_that_check(tmp_path):
    chunks = _chunks(tmp_path, 1)
    path = _vectors(tmp_path, [[1.0, 0.0]], chunks, count=1, dim=2, dtype="float32", sha="")
    vectors, meta = load_vectors(path, chunks_file=chunks, chunks_sha1=file_sha1(chunks))
    assert meta.chunks_sha1 == ""
    assert vectors.shape == (1, 2)


def test_sidecar_carries_unknown_fields_through():
    meta = VectorMeta.from_json({"model": "m", "dim": 2, "count": 1, "future_field": [1, 2]})
    assert meta.to_json()["future_field"] == [1, 2]


def test_count_lines_counts_newlines(tmp_path):
    chunks = _chunks(tmp_path, 4)
    assert count_lines(chunks) == 4
