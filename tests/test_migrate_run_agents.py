"""run 目录改名：A/B/C → perceiver/writer/reviewer。"""

import importlib.util
import json
from pathlib import Path

from helpers import phrase, write_song

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "migrate_run_agents.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_run_agents", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _load_script()


def _make_run(base: Path, run_id: str = "run-1") -> Path:
    run = base / run_id
    (run / "A").mkdir(parents=True)
    (run / "A" / "output.json").write_text('{"theme": "x"}', encoding="utf-8")
    (run / "A" / "meta.json").write_text(json.dumps({"agent": "A", "model": "m"}), encoding="utf-8")
    for index, letter in ((1, "B"), (2, "C")):
        d = run / "sections" / f"{index:03d}" / letter
        d.mkdir(parents=True)
        (d / "output.json").write_text("{}", encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"agent": letter}), encoding="utf-8")
    (run / "lyrics.zh.txt").write_text("词", encoding="utf-8")
    return run


def test_migrate_renames_and_retags(tmp_path):
    root = tmp_path / "songs"
    run = _make_run(root / "t" / "workflow")

    assert mod.main(["--root", str(root), "--song", "t"]) == 0

    assert (run / "perceiver" / "output.json").is_file()
    assert (run / "sections" / "001" / "writer" / "output.json").is_file()
    assert (run / "sections" / "002" / "reviewer" / "output.json").is_file()
    assert not (run / "A").exists() and not (run / "sections" / "001" / "B").exists()
    # meta.json 里的 agent 字段跟着改，别留下自相矛盾的记录
    assert json.loads((run / "perceiver" / "meta.json").read_text(encoding="utf-8"))["agent"] == "perceiver"
    assert json.loads((run / "sections" / "001" / "writer" / "meta.json").read_text(encoding="utf-8"))["agent"] == "writer"
    assert json.loads((run / "perceiver" / "meta.json").read_text(encoding="utf-8"))["model"] == "m"
    # 其它文件原样不动
    assert (run / "lyrics.zh.txt").is_file()


def test_dry_run_does_not_touch_disk(tmp_path, capsys):
    root = tmp_path / "songs"
    run = _make_run(root / "t" / "workflow")

    assert mod.main(["--root", str(root), "--song", "t", "--dry-run"]) == 0

    assert (run / "A").is_dir()
    assert not (run / "perceiver").exists()
    out = capsys.readouterr().out
    assert "将改" in out
    assert "去掉 --dry-run" in out


def test_second_run_is_idempotent(tmp_path):
    root = tmp_path / "songs"
    run = _make_run(root / "t" / "workflow")

    assert mod.main(["--root", str(root), "--song", "t"]) == 0
    assert mod.main(["--root", str(root), "--song", "t"]) == 0  # 再来一次不炸

    assert (run / "perceiver").is_dir()
    assert json.loads((run / "sections" / "001" / "writer" / "meta.json").read_text(encoding="utf-8"))["agent"] == "writer"


def test_preview_directories_are_not_touched(tmp_path):
    root = tmp_path / "songs"
    _make_run(root / "t" / "workflow")
    preview = root / "t" / "workflow" / "preview-1"
    (preview / "A").mkdir(parents=True)

    assert mod.main(["--root", str(root), "--song", "t"]) == 0

    assert (preview / "A").is_dir()  # 请求快照不改


def test_baselines_need_the_flag(tmp_path):
    root = tmp_path / "songs"
    base = root / "t" / "baselines" / "one-shot"
    run = _make_run(base)

    assert mod.main(["--root", str(root), "--song", "t"]) == 0
    assert (run / "A").is_dir()  # 默认不动 baseline

    assert mod.main(["--root", str(root), "--song", "t", "--include-baselines"]) == 0
    assert (run / "perceiver").is_dir()
    assert (run / "sections" / "001" / "writer").is_dir()


def test_all_songs_walks_the_library(tmp_path):
    root = tmp_path / "songs"
    write_song(root, "s1", ["S1"], [phrase("A-01", "S1", "ああ", 2)])
    write_song(root, "s2", ["S1"], [phrase("A-01", "S1", "ああ", 2)])
    run1 = _make_run(root / "s1" / "workflow")
    run2 = _make_run(root / "s2" / "workflow")

    assert mod.main(["--root", str(root), "--all"]) == 0

    assert (run1 / "perceiver").is_dir()
    assert (run2 / "perceiver").is_dir()


def test_empty_library_is_an_error(tmp_path, capsys):
    root = tmp_path / "songs"
    root.mkdir()
    assert mod.main(["--root", str(root), "--all"]) == 1
    assert "歌库是空的" in capsys.readouterr().err


def test_plan_moves_only_lists_existing_directories(tmp_path):
    run = _make_run(tmp_path / "workflow")
    (run / "sections" / "003").mkdir()  # 空段号

    pairs = {(src.name, dst.name) for src, dst in mod.plan_moves(run)}
    assert pairs == {("A", "perceiver"), ("B", "writer"), ("C", "reviewer")}
