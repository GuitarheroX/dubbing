"""可选本地 COMET-QE；未显式启用时不导入或下载模型。"""

from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import math
from numbers import Real

from .common import VARIANTS, average, digest, file_hash, read, write, variants_of

DEFAULT_MODEL = "Unbabel/wmt22-cometkiwi-da"


def build_records(snapshot):
    records = []
    for item in snapshot["songs"]:
        for variant in variants_of(snapshot):
            texts = item["candidates"][variant]["texts"]
            for phrase in item["song"]["phrases"]:
                if not phrase.get("keep_source") and texts.get(phrase["id"]):
                    records.append({"song": item["directory"], "section": phrase["section"],
                                    "id": phrase["id"], "variant": variant,
                                    "src": phrase["ja"], "mt": texts[phrase["id"]]})
    return records


def local_predict(records, *, model_name, revision, checkpoint, local_only, batch_size, gpus, seed):
    try:
        import torch
        from comet import load_from_checkpoint
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError('COMET-QE 需要可选依赖：python -m pip install -e ".[evaluation]"') from exc
    if checkpoint:
        checkpoint_path = Path(checkpoint).resolve()
        resolved_revision = None
    else:
        model_path = Path(snapshot_download(repo_id=model_name, revision=revision or "main",
                                            allow_patterns=["checkpoints/model.ckpt", "hparams.yaml"],
                                            local_files_only=local_only))
        checkpoint_path = model_path / "checkpoints/model.ckpt"
        resolved_revision = model_path.name
    hparams = checkpoint_path.parent.parent / "hparams.yaml"
    if not checkpoint_path.is_file() or not hparams.is_file():
        raise ValueError("COMET 模型缺少 checkpoints/model.ckpt 或根目录 hparams.yaml")
    torch.manual_seed(seed)
    model = load_from_checkpoint(str(checkpoint_path), local_files_only=local_only)
    model.eval()
    prediction = model.predict([{"src": r["src"], "mt": r["mt"]} for r in records], batch_size=batch_size, gpus=gpus)
    scores = prediction.scores if hasattr(prediction, "scores") else prediction["scores"]
    return scores, {"resolved_revision": resolved_revision, "checkpoint_sha256": file_hash(checkpoint_path),
                    "hparams_sha256": file_hash(hparams),
                    "encoder_revision": getattr(getattr(getattr(getattr(model, "encoder", None), "model", None), "config", None), "_commit_hash", None),
                    "versions": {p: version(p) for p in ("unbabel-comet", "torch", "transformers", "huggingface-hub")}}


def aggregate(records, songs, variants=VARIANTS):
    result = {}
    for variant in variants:
        by_song = defaultdict(list)
        for row in records:
            if row["variant"] == variant and row["song"] in songs:
                by_song[row["song"]].append(row["score"])
        result[variant] = {"macro_score": average([average(v) for v in by_song.values()]),
                           "valid_songs": len(by_song), "phrases": sum(len(v) for v in by_song.values())}
    return result


def run_comet(snapshot, rules, out, *, model_name=DEFAULT_MODEL, revision=None, checkpoint=None,
              local_only=False, batch_size=8, gpus=0, seed=0, predictor=None):
    out = Path(out)
    config = {"input_hash": snapshot["input_hash"], "model": model_name, "revision_requested": revision,
              "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
              "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
              "local_only": local_only, "batch_size": batch_size, "gpus": gpus, "seed": seed}
    if out.exists():
        previous = read(out)
        if previous["config"] != config:
            raise ValueError("COMET 设置或输入已变化，请使用新的评测目录")
        if previous.get("status") == "completed":
            if digest(previous["phrases"]) != previous.get("records_hash"):
                raise ValueError("已有 COMET 分数记录被修改，不能复用")
            return previous
    rows = build_records(snapshot)
    if not rows:
        result = {"status": "not_evaluated", "config": config, "reason": "没有可对齐的需译乐句"}
        write(out, result)
        return result
    scores, metadata = (predictor or local_predict)(rows, model_name=model_name, revision=revision,
                                                  checkpoint=checkpoint, local_only=local_only,
                                                  batch_size=batch_size, gpus=gpus, seed=seed)
    scores = list(scores)
    if len(scores) != len(rows):
        raise ValueError("COMET 返回分数数量与输入乐句数不一致")
    for row, score in zip(rows, scores, strict=True):
        if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
            raise ValueError("COMET 返回了非有限数值分数")
        row["score"] = float(score)
    result = {"status": "completed", "config": config, "model_metadata": metadata,
              "phrases": rows, "records_hash": digest(rows), "all_available": aggregate(rows, {s["directory"] for s in snapshot["songs"]}, variants_of(snapshot)),
              "paired": aggregate(rows, rules["paired_songs"], variants_of(snapshot)),
              "note": "通用翻译质量诊断；未读取旋律、预算或音频，不代表实际可唱性。"}
    write(out, result)
    return result
