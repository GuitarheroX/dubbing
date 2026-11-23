"""python -m evaluate：默认仅离线评测，按需启用模型阶段。"""

import argparse
from pathlib import Path
import re
import subprocess
from uuid import uuid4

from dubbing.llm import load_dotenv
from dubbing.recorded_client import RecordedAPIClient

from . import human
from .common import AVAILABLE_VARIANTS, REPO_ROOT, VARIANTS, digest, file_hash, now, read, write, variants_of
from .comet_qe import DEFAULT_MODEL, run_comet
from .inputs import collect
from .judges import run_judges
from .report import publish
from .rules import evaluate_rules


def code_hashes():
    paths = list(Path(__file__).parent.glob("*.py")) + [REPO_ROOT / "dubbing" / name for name in ("checks.py", "counting.py", "models.py", "recorded_client.py", "llm.py")]
    return {str(p.relative_to(REPO_ROOT)): file_hash(p) for p in paths}


def parser():
    p = argparse.ArgumentParser(description="译配结果评测，默认四组，可加入 one-shot 消融；默认离线，不调用模型")
    p.add_argument("--variant", action="append", choices=AVAILABLE_VARIANTS, help="选取评测组，可重复；默认原有四组，五组比较时显式列出五组")
    p.add_argument("--root", type=Path, help="四组共同歌库；默认优先 data/songs_rag，否则 data/songs")
    p.add_argument("--workflow-root", type=Path, help="无 RAG workflow 所在歌库，默认与 --root 相同")
    p.add_argument("--rag-root", type=Path, help="RAG workflow 所在歌库，默认与 --root 相同")
    p.add_argument("--baseline-root", type=Path, help="论文 baseline 与 one-shot 所在歌库，默认与 --root 相同")
    p.add_argument("--song", action="append", help="目录名，可重复；默认全库")
    p.add_argument("--batch", type=Path, help="固定 baseline 批次的 final-summary.json；默认冻结各组 latest")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--out", type=Path, help="新建评测目录，不覆盖现有目录")
    target.add_argument("--resume", type=Path, help="复用冻结输入，续跑评审或补充 COMET / 真人数据")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    judge = p.add_mutually_exclusive_group()
    judge.add_argument("--judge", action="store_true", help="启用所选组的文本评分与所有候选对的顺序复评")
    judge.add_argument("--judge-dry-run", action="store_true", help="只写匿名评审请求，不调用 API")
    p.add_argument("--client", choices=("pending", "api"), default="pending")
    p.add_argument("--judge-name", default="judge-1", help="评审模型记录名称；第二个模型用不同名称")
    p.add_argument("--reuse-judge", type=Path, help="复用另一 judge 目录中完整请求及配置一致的已完成任务")
    p.add_argument("--judge-limit", type=int, help="仅选前 N 个共同完成的段落，适合先预览")
    p.add_argument("--judge-jobs", type=int, default=1, help="并发评审数，每个任务独立客户端；默认 1")
    p.add_argument("--model", help="评审模型，默认 .env 的 DUBBING_MODEL")
    p.add_argument("--base-url", help="评审接口，默认 .env 的 DUBBING_BASE_URL")
    p.add_argument("--protocol", choices=("anthropic", "openai"), default="anthropic")
    p.add_argument("--temperature", type=float, default=0, help="评审温度，默认 0；不保证完全确定")
    p.add_argument("--thinking", choices=("disabled", "enabled"), default="disabled")
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--comet", action="store_true", help="显式启用本地 COMET-QE，可能下载模型")
    p.add_argument("--comet-model", default=DEFAULT_MODEL)
    p.add_argument("--comet-revision", help="Hugging Face commit / revision，记录实际解析版本")
    p.add_argument("--comet-checkpoint", type=Path, help="使用已下载的 checkpoints/model.ckpt")
    p.add_argument("--comet-local-only", action="store_true", help="COMET 及编码器只读取本地缓存")
    p.add_argument("--comet-batch-size", type=int, default=8)
    p.add_argument("--comet-gpus", type=int, default=0, help="默认 CPU")
    p.add_argument("--human-ratings", type=Path, help="导入真人演唱评分 JSON")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.judge and args.client != "api":
        p.error("真实评审需要 --judge --client api；预览用 --judge-dry-run")
    if args.resume and (args.root or args.song or args.batch or args.workflow_root or args.rag_root or args.baseline_root or args.variant):
        p.error("--resume 使用已冻结输入，不能同时重新指定歌库或批次")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.judge_name):
        p.error("--judge-name 只能包含字母、数字、下划线和连字符")
    if any(v < 1 for v in (args.bootstrap_samples, args.max_tokens, args.max_retries, args.comet_batch_size, args.judge_jobs)) or args.comet_gpus < 0 or args.judge_limit is not None and args.judge_limit < 1:
        p.error("样本数、token、重试、batch size 和段落上限必须为正，GPU 数不能为负")
    load_dotenv(REPO_ROOT / ".env")
    directory = None
    active = False
    snapshot = rules = None
    try:
        if args.resume:
            directory = args.resume.resolve()
            manifest = read(directory / "manifest.json")
            snapshot = read(directory / "inputs.json")
            variants_of(snapshot)
            if manifest.get("schema_version") != 2:
                raise ValueError("评测快照版本不兼容，请新建评测目录")
            if digest(snapshot["songs"]) != snapshot["input_hash"] or digest(snapshot) != manifest["snapshot_hash"]:
                raise ValueError("评测输入快照已被修改")
            if manifest["code_hashes"] != code_hashes():
                raise ValueError("评测代码已变化，请新建评测目录")
            if args.seed != manifest["seed"] or args.bootstrap_samples != manifest["bootstrap_samples"]:
                raise ValueError("续跑随机种子或 bootstrap 设置不同")
            active = True
        else:
            default_root = REPO_ROOT / "data/songs_rag"
            if not any(default_root.glob("*/song.yaml")):
                default_root = REPO_ROOT / "data/songs"
            snapshot = collect(args.root or default_root, args.song, args.batch,
                               workflow_root=args.workflow_root, rag_root=args.rag_root, baseline_root=args.baseline_root,
                               variants=args.variant or VARIANTS)
            directory = (args.out or REPO_ROOT / "data/evaluations" / ("eval-" + now().replace(":", "").replace("+", "-") + "-" + uuid4().hex[:8])).resolve()
            directory.mkdir(parents=True, exist_ok=False)
            active = True
            code = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
            manifest = {"schema_version": 2, "created_at": now(), "snapshot_hash": digest(snapshot),
                        "code_commit": code.stdout.strip() if code.returncode == 0 else None,
                        "code_hashes": code_hashes(), "seed": args.seed, "bootstrap_samples": args.bootstrap_samples}
            write(directory / "manifest.json", manifest)
            write(directory / "inputs.json", snapshot)
            write(directory / "human-template.json", human.template())
        write(directory / "meta.json", {"status": "running", "updated_at": now()})
        rules = evaluate_rules(snapshot, seed=args.seed, samples=args.bootstrap_samples)
        write(directory / "rules.json", rules)
        publish(directory, snapshot, rules)
        partial = False
        if args.judge or args.judge_dry_run:
            def create_client():
                return RecordedAPIClient(model=args.model, base_url=args.base_url, protocol=args.protocol,
                                         thinking=args.thinking, temperature=args.temperature,
                                         max_tokens=args.max_tokens, max_retries=args.max_retries)
            client = create_client()
            if args.judge and not client.api_key:
                raise ValueError("未配置评审 API key")
            result = run_judges(snapshot, rules, client, directory / "judges" / args.judge_name,
                                seed=args.seed, limit=args.judge_limit, dry_run=args.judge_dry_run,
                                jobs=args.judge_jobs, client_factory=create_client, reuse_from=args.reuse_judge)
            partial = result["failed_tasks"] > 0
        if args.comet:
            run_comet(snapshot, rules, directory / "comet.json", model_name=args.comet_model,
                      revision=args.comet_revision, checkpoint=args.comet_checkpoint,
                      local_only=args.comet_local_only, batch_size=args.comet_batch_size,
                      gpus=args.comet_gpus, seed=args.seed)
        if args.human_ratings:
            ratings = human.load_ratings(args.human_ratings, snapshot)
            ratings["input_hash"] = snapshot["input_hash"]
            write(directory / "human.json", ratings)
        partial = partial or any(read(f).get("status") == "partial" for f in (directory / "judges").glob("*/summary.json"))
        write(directory / "meta.json", {"status": "partial" if partial else "completed", "updated_at": now()})
        publish(directory, snapshot, rules)
        print(f"评测完成：{directory / 'report.md'}；共同完成 {len(rules['paired_songs'])}/{len(snapshot['songs'])} 首。")
        return 1 if partial else 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        # 运行错误不回写候选译词；保留已成功的离线 / 评审记录供续跑。
        message = str(exc)
        if active and directory is not None and snapshot is not None and (directory / "manifest.json").exists():
            write(directory / "meta.json", {"status": "failed", "error": message, "updated_at": now()})
            if rules is not None:
                publish(directory, snapshot, rules)
        print(f"评测失败：{message}")
        return 2
