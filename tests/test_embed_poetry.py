"""服务器脚本：chunks.jsonl → vectors.npy（编码器用替身，不需要 GPU）。"""

import importlib.util
import json
from pathlib import Path

import pytest

from dubbing.rag.vectors import VectorMeta, load_vectors, sidecar_path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "embed_poetry.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("embed_poetry", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script()


def _fake_encoder(dim=4):
    def encode(texts):
        out = []
        for text in texts:
            vector = [0.0] * dim
            vector[len(text) % dim] = 1.0
            out.append(vector)
        return out

    return encode


def _write_chunks(tmp_path, n=5):
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(
                json.dumps(
                    {"chunk_id": f"p/诗#{i}", "text": f"第{i}节诗句", "title": "诗"},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def test_count_and_hash_matches_sha1_and_lines(tmp_path):
    chunks = _write_chunks(tmp_path, 3)
    lines, sha1 = mod.count_and_hash(chunks)
    assert lines == 3
    assert len(sha1) == 40


def test_embed_chunks_keeps_row_order_and_dtype():
    rows = [{"text": "甲"}, {"text": "乙乙"}, {"text": "丙丙丙"}]
    vectors = mod.embed_chunks(rows, _fake_encoder(dim=3), dtype="float16", batch_size=2)

    assert vectors.shape == (3, 3)
    assert str(vectors.dtype) == "float16"
    starts = [int(v.argmax()) for v in vectors]
    assert starts == [1 % 3, 2 % 3, 3 % 3]


def test_embed_chunks_rejects_a_bad_encoder():
    with pytest.raises(Exception, match="二维矩阵"):
        mod.embed_chunks([{"text": "甲"}], lambda texts: [1.0, 2.0])


def test_main_writes_vectors_and_sidecar(tmp_path, monkeypatch, capsys):
    chunks = _write_chunks(tmp_path, 4)
    out = tmp_path / "vectors.npy"
    monkeypatch.setattr(mod, "build_encoder", lambda *a, **k: _fake_encoder(dim=4))

    code = mod.main(
        [
            "--chunks",
            str(chunks),
            "--out",
            str(out),
            "--model",
            "stub-model",
            "--batch",
            "2",
            "--quiet",
        ]
    )

    assert code == 0
    vectors, meta = load_vectors(out, expected_count=4, chunks_file=chunks)
    assert vectors.shape == (4, 4)
    assert meta.model == "stub-model"
    assert meta.count == 4
    assert meta.dtype == "float16"
    assert meta.normalize is True
    assert meta.pooling == "cls"
    assert meta.chunks_sha1  # 记下了 chunks 的指纹
    assert sidecar_path(out).is_file()
    err = capsys.readouterr().err
    assert "完成：4 条 × 4 维" in err
    assert "build_poetry_index.py --vectors" in err


def test_main_can_emit_float32(tmp_path, monkeypatch):
    chunks = _write_chunks(tmp_path, 2)
    out = tmp_path / "vectors.npy"
    monkeypatch.setattr(mod, "build_encoder", lambda *a, **k: _fake_encoder(dim=2))

    assert mod.main(["--chunks", str(chunks), "--out", str(out), "--dtype", "float32", "--quiet"]) == 0
    _, meta = load_vectors(out, expected_count=2)
    assert meta.dtype == "float32"


def test_limit_writes_a_matching_subset_of_chunks(tmp_path, monkeypatch, capsys):
    """--limit 时 chunks 也要跟着截断，否则向量和 chunks 对不上。"""
    chunks = _write_chunks(tmp_path, 6)
    out = tmp_path / "vectors.head.npy"
    seen: list[int] = []

    def encoder(texts):
        seen.append(len(texts))
        return _fake_encoder(dim=2)(texts)

    monkeypatch.setattr(mod, "build_encoder", lambda *a, **k: encoder)

    assert mod.main(["--chunks", str(chunks), "--out", str(out), "--limit", "3", "--quiet"]) == 0

    subset = chunks.with_name("chunks.limit3.jsonl")
    assert subset.is_file()
    vectors, meta = load_vectors(out, expected_count=3, chunks_file=subset)
    assert vectors.shape[0] == 3
    assert meta.chunks_file == str(subset)
    assert sum(seen) == 3  # 只编码了 3 条
    assert "--limit 3" in capsys.readouterr().err


def test_missing_chunks_is_reported(tmp_path, capsys):
    assert mod.main(["--chunks", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "v.npy")]) == 1
    assert "找不到切块文件" in capsys.readouterr().err


def test_sidecar_records_what_the_server_used(tmp_path, monkeypatch):
    chunks = _write_chunks(tmp_path, 1)
    out = tmp_path / "vectors.npy"
    monkeypatch.setattr(mod, "build_encoder", lambda *a, **k: _fake_encoder(dim=8))

    assert (
        mod.main(
            [
                "--chunks",
                str(chunks),
                "--out",
                str(out),
                "--model",
                "/data/models/bge-large-zh-v1.5",
                "--device",
                "cuda",
                "--max-length",
                "256",
                "--quiet",
            ]
        )
        == 0
    )
    raw = json.loads(sidecar_path(out).read_text(encoding="utf-8"))
    assert raw["model"] == "/data/models/bge-large-zh-v1.5"
    assert raw["device"] == "cuda"
    assert raw["max_length"] == 256
    assert raw["dim"] == 8
    assert raw["created_at"]
    assert VectorMeta.from_json(raw).count == 1
