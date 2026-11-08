"""建库脚本：语料 → 切块 → （可选的）嵌入与本地 Chroma。

两条路都测：在线（边切边调嵌入端点）与离线（服务器给向量，本机只灌库）。
嵌入、向量与向量库都用替身，不联网；没装 chromadb 时只跑 --dry-run 与 --vectors 的报错分支。
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from dubbing.rag.chunking import count_tokens_by_chars

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_poetry_index.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_poetry_index", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script()
#: 在 autouse fixture 打补丁之前抓住真函数，供测 tokenizer 回退用。
REAL_LOAD_TOKEN_COUNTER = mod.load_token_counter

RAIN = """title:雨
date:20030508

雨水发着微小的酸
雨水这块化石油

雨水平行黑色黄的迂回
雨水确知大空的疮痍
"""


def _corpus(tmp_path: Path) -> Path:
    """一个极小语料：1 首正常（2 节）+ 1 首完全重复 + 1 首太短。"""
    data = tmp_path / "raw" / "data"
    poet = data / "00913_00913"
    poet.mkdir(parents=True)
    (poet / "a雨.pt").write_text(RAIN, encoding="utf-8")
    (poet / "b重复.pt").write_text(RAIN.replace("title:雨", "title:转载"), encoding="utf-8")
    (poet / "c太短.pt").write_text("title:太短\n\n太短", encoding="utf-8")
    return data.parent


@pytest.fixture(autouse=True)
def _no_tokenizer_download(monkeypatch):
    """别在测试里下载真 tokenizer。"""
    monkeypatch.setattr(mod, "load_token_counter", lambda model: (count_tokens_by_chars, "chars(test)"))


def test_dry_run_writes_chunks_and_stats(tmp_path, capsys):
    root = _corpus(tmp_path)
    out = tmp_path / "index"

    assert mod.main(["--poetry-root", str(root), "--out", str(out), "--dry-run", "--quiet"]) == 0

    rows = [json.loads(line) for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["chunk_id"] for r in rows] == ["00913_00913/a雨#s1", "00913_00913/a雨#s2"]
    assert rows[0]["text"] == "雨水发着微小的酸\n雨水这块化石油"
    assert rows[0]["stanza_idx"] == 1
    assert rows[0]["chars"] == 15

    err = capsys.readouterr().err
    assert "跳过 1 首" in err
    assert "去重 1 首" in err
    assert "index_id 会是" in err
    assert not (out / "chroma").exists()  # dry-run 不碰向量库


def test_dry_run_rejects_a_missing_corpus(tmp_path, capsys):
    assert mod.main(["--poetry-root", str(tmp_path / "nope"), "--dry-run"]) == 1
    assert "找不到语料目录" in capsys.readouterr().err


def test_dry_run_rejects_a_directory_without_poems(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mod.main(["--poetry-root", str(empty), "--dry-run"]) == 1
    assert "没有 <诗人>/<诗>.pt" in capsys.readouterr().err


def test_full_build_writes_chroma_and_manifest(tmp_path, capsys, monkeypatch):
    pytest.importorskip("chromadb")
    from dubbing.rag.store import ChromaStore, read_manifest

    root = _corpus(tmp_path)
    out = tmp_path / "index"

    class FakeEmbedding:
        name = "fake"
        model = "fake-3dim"

        def __init__(self):
            self.batches: list[list[str]] = []

        def embed(self, texts):
            self.batches.append(list(texts))
            return [[1.0, 0.0, 0.0] for _ in texts]

    fake = FakeEmbedding()
    monkeypatch.setattr(mod, "build_embedding_client", lambda **kwargs: fake)

    assert mod.main(["--poetry-root", str(root), "--out", str(out), "--quiet", "--batch", "1"]) == 0

    store = ChromaStore(out / "chroma", "poetry_stanza256_v1")
    assert store.count() == 2
    hits = store.query([1.0, 0.0, 0.0], 1)
    assert hits[0].title == "雨"
    assert hits[0].poet == "00913"

    manifest = read_manifest(out / "chroma")
    assert manifest["corpus"] == {
        "path": str(root / "data"),
        "poems": 1,
        "skipped": 1,
        "duplicates": 1,
        "chunks": 2,
    }
    assert manifest["chunking"] == {
        "unit": "stanza",
        "max_tokens": 256,
        "overlap": 0,
        "tokenizer": "chars(test)",
    }
    assert manifest["model"]["name"] == "fake-3dim"
    assert manifest["model"]["dim"] == 3
    assert manifest["model"]["instruction"] is None
    assert manifest["stored"] == 2
    assert manifest["index_id"] == "poetry_stanza256_fake-3dim_v1"
    assert "完成" in capsys.readouterr().err


def test_build_can_override_the_index_id(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from dubbing.rag.store import read_manifest

    root = _corpus(tmp_path)
    out = tmp_path / "index"

    class FakeEmbedding:
        name = "fake"
        model = "m"

        def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(mod, "build_embedding_client", lambda **kwargs: FakeEmbedding())
    assert mod.main(
        [
            "--poetry-root",
            str(root),
            "--out",
            str(out),
            "--quiet",
            "--index-id",
            "my-index-v9",
        ]
    ) == 0
    assert read_manifest(out / "chroma")["index_id"] == "my-index-v9"


def test_load_token_counter_falls_back_without_transformers(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "transformers", None)  # import 会失败
    counter, name = REAL_LOAD_TOKEN_COUNTER("BAAI/bge-large-zh-v1.5")

    assert name == "chars(fallback)"
    assert counter("一二 三") == 3
    assert "改用" in capsys.readouterr().err


def test_load_token_counter_falls_back_when_the_model_is_unavailable(monkeypatch, capsys):
    fake = types.ModuleType("transformers")

    def boom(*args, **kwargs):
        raise OSError("can't reach HuggingFace")

    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=boom)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)

    counter, name = REAL_LOAD_TOKEN_COUNTER("自建模型-不在-HF")

    assert name == "chars(fallback)"
    assert counter("一二三四五六七八九十") == 10
    assert "改用" in capsys.readouterr().err


def test_long_stanza_is_split_during_the_build(tmp_path):
    data = tmp_path / "raw" / "data" / "p_x"
    data.mkdir(parents=True)
    (data / "长.pt").write_text("title:长\n\n" + "一二三四五六七八九十\n" * 6, encoding="utf-8")
    out = tmp_path / "index"

    assert mod.main(
        ["--poetry-root", str(tmp_path / "raw"), "--out", str(out), "--dry-run", "--max-tokens", "25", "--quiet"]
    ) == 0

    rows = [json.loads(line) for line in (out / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3  # 6 行 × 10 字，每块最多 25 token → 2 行一块
    assert [r["part_idx"] for r in rows] == [1, 2, 3]
    assert all(r["n_parts"] == 3 for r in rows)
    assert all(r["tokens"] <= 25 for r in rows)


# -- 离线向量：服务器给向量，本机只灌库 ----------------------------------


def _write_chunks_jsonl(path: Path, count: int) -> Path:
    """手写一份 chunks.jsonl（字段与 Chunk 一致），免得依赖切块细节。"""
    with path.open("w", encoding="utf-8") as fh:
        for i in range(1, count + 1):
            row = {
                "chunk_id": f"p_x/诗#{i}",
                "poem_id": "p_x/诗",
                "poet": "p",
                "title": "诗",
                "date": "",
                "stanza_idx": i,
                "part_idx": 1,
                "n_parts": 1,
                "text": f"第{i}节诗句",
                "chars": 5,
                "tokens": 5,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _make_vectors_file(chunks: Path, vectors_path: Path, rows, *, dim, model="stub-model", sha=None):
    from dubbing.rag.vectors import VectorMeta, file_sha1, save_vectors

    meta = VectorMeta(
        model=model,
        dim=dim,
        count=len(rows),
        dtype="float32",
        chunks_file=str(chunks),
        chunks_sha1=sha if sha is not None else file_sha1(chunks),
    )
    save_vectors(vectors_path, rows, meta)
    return vectors_path


def test_offline_vectors_are_ingested_without_any_embedding_service(tmp_path):
    pytest.importorskip("chromadb")
    from dubbing.rag.store import ChromaStore, read_manifest

    chunks = _write_chunks_jsonl(tmp_path / "chunks.jsonl", 4)
    vectors_path = _make_vectors_file(
        chunks,
        tmp_path / "vectors.npy",
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        dim=3,
    )
    out = tmp_path / "index"

    assert (
        mod.main(
            [
                "--vectors",
                str(vectors_path),
                "--chunks",
                str(chunks),
                "--out",
                str(out),
                "--quiet",
            ]
        )
        == 0
    )

    store = ChromaStore(out / "chroma", "poetry_stanza256_v1")
    assert store.count() == 4
    hit = store.query([1.0, 0.0, 0.0], 1)[0]
    assert hit.chunk_id in {"p_x/诗#1", "p_x/诗#4"}   # 余弦相同，取到哪条都算对
    assert hit.title == "诗" and hit.poet == "p"

    manifest = read_manifest(out / "chroma")
    assert manifest["source"] == "vectors"
    assert manifest["index_id"] == "poetry_stanza256_stub-model_v1"
    assert manifest["model"] == {
        "name": "stub-model",
        "dim": 3,
        "pooling": "cls",
        "normalize": True,
        "instruction": None,
    }
    assert manifest["corpus"]["chunks"] == 4
    assert manifest["corpus"]["sha1"]
    assert manifest["chunking"]["tokenizer"] == "unknown(offline)"
    assert manifest["stored"] == 4


def test_offline_vectors_respect_batch_size(tmp_path):
    pytest.importorskip("chromadb")
    from dubbing.rag.store import ChromaStore

    chunks = _write_chunks_jsonl(tmp_path / "chunks.jsonl", 5)
    vectors_path = _make_vectors_file(
        chunks,
        tmp_path / "vectors.npy",
        [[float(i), 0.0] for i in range(1, 6)],
        dim=2,
    )
    out = tmp_path / "index"

    assert (
        mod.main(
            [
                "--vectors",
                str(vectors_path),
                "--chunks",
                str(chunks),
                "--out",
                str(out),
                "--batch",
                "2",
                "--quiet",
            ]
        )
        == 0
    )
    assert ChromaStore(out / "chroma", "poetry_stanza256_v1").count() == 5


def test_offline_mode_reports_a_mismatched_chunks_file(tmp_path, capsys):
    from dubbing.rag.store import ChromaStore

    chunks = _write_chunks_jsonl(tmp_path / "chunks.jsonl", 3)
    vectors_path = _make_vectors_file(
        chunks, tmp_path / "vectors.npy", [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dim=2
    )
    # 切块文件被改过（行数不变，内容变了）→ sha1 对不上
    text = chunks.read_text(encoding="utf-8").replace("第1节诗句", "改过的第一节")
    chunks.write_text(text, encoding="utf-8")

    out = tmp_path / "index"
    code = mod.main(
        ["--vectors", str(vectors_path), "--chunks", str(chunks), "--out", str(out), "--quiet"]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "向量文件不可用" in err
    assert "对不上" in err
    # 校验没过就不该写进任何向量（Chroma 目录会被建出来，但 collection 是空的）
    assert ChromaStore(out / "chroma", "poetry_stanza256_v1").count() == 0


def test_offline_mode_reports_a_count_mismatch(tmp_path, capsys):
    chunks = _write_chunks_jsonl(tmp_path / "chunks.jsonl", 3)
    vectors_path = _make_vectors_file(
        chunks, tmp_path / "vectors.npy", [[1.0, 0.0], [0.0, 1.0]], dim=2
    )  # 少了 1 条
    out = tmp_path / "index"

    assert (
        mod.main(
            ["--vectors", str(vectors_path), "--chunks", str(chunks), "--out", str(out), "--quiet"]
        )
        == 1
    )
    assert "向量文件不可用" in capsys.readouterr().err


def test_offline_mode_needs_the_chunks_file(tmp_path, capsys):
    vectors_path = tmp_path / "vectors.npy"
    vectors_path.write_bytes(b"")
    assert (
        mod.main(
            [
                "--vectors",
                str(vectors_path),
                "--chunks",
                str(tmp_path / "nope.jsonl"),
                "--out",
                str(tmp_path / "index"),
            ]
        )
        == 1
    )
    assert "找不到切块文件" in capsys.readouterr().err


def test_offline_mode_rejects_dry_run(tmp_path, capsys):
    assert mod.main(["--vectors", str(tmp_path / "v.npy"), "--dry-run"]) == 2
    assert "--dry-run 是给切块用的" in capsys.readouterr().err


def test_verify_accepts_matching_vectors(tmp_path, monkeypatch, capsys):
    pytest.importorskip("chromadb")
    chunks = _write_chunks_jsonl(tmp_path / "chunks.jsonl", 3)
    vectors_path = _make_vectors_file(
        chunks, tmp_path / "vectors.npy", [[1.0, 0.0, 0.0]] * 3, dim=3
    )

    class FakeLocal:
        def __init__(self, model, device=None, batch_size=64):
            self.model = model

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr("dubbing.rag.embedding.LocalEmbeddingClient", FakeLocal)
    out = tmp_path / "index"

    code = mod.main(
        [
            "--vectors",
            str(vectors_path),
            "--chunks",
            str(chunks),
            "--out",
            str(out),
            "--verify",
            "2",
            "--quiet",
        ]
    )

    assert code == 0
    err = capsys.readouterr().err
    assert "抽查 2 条" in err
    assert "一致" in err


def test_verify_flags_inconsistent_vectors(tmp_path, monkeypatch, capsys):
    pytest.importorskip("chromadb")
    chunks = _write_chunks_jsonl(tmp_path / "chunks.jsonl", 3)
    vectors_path = _make_vectors_file(
        chunks, tmp_path / "vectors.npy", [[1.0, 0.0, 0.0]] * 3, dim=3
    )

    class WrongLocal:
        def __init__(self, model, device=None, batch_size=64):
            self.model = model

        def embed(self, texts):
            return [[0.0, 1.0, 0.0] for _ in texts]   # 与文件里的向量完全不同

    monkeypatch.setattr("dubbing.rag.embedding.LocalEmbeddingClient", WrongLocal)
    out = tmp_path / "index"

    code = mod.main(
        [
            "--vectors",
            str(vectors_path),
            "--chunks",
            str(chunks),
            "--out",
            str(out),
            "--verify",
            "1",
            "--quiet",
        ]
    )

    assert code == 1
    assert "警告" in capsys.readouterr().err
