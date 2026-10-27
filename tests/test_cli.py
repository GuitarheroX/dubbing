from dubbing.cli import main

from helpers import phrase, write_song


def _song(root):
    write_song(
        root,
        "t",
        ["S1"],
        [phrase("A-01", "S1", "ああああ", 4), phrase("A-02", "S1", "らら", keep_source=True)],
        title="测试歌",
    )


def test_songs_lists_budget(root, capsys):
    _song(root)
    assert main(["--root", str(root), "songs"]) == 0
    out = capsys.readouterr().out
    assert "测试歌" in out
    assert "1 段 / 2 句 / 预算共 4 字" in out


def test_dry_run_writes_preview(root, capsys):
    _song(root)
    assert main(["--root", str(root), "run", "--song", "t", "--dry-run"]) == 0

    preview = root / "t" / "preview"
    assert (preview / "a.system.md").is_file()
    assert (preview / "a.data.json").is_file()
    assert (preview / "b-S1.data.json").is_file()
    assert "没有调用任何模型" in capsys.readouterr().out


def test_view_shows_pending_and_skip(root, capsys):
    _song(root)
    assert main(["--root", str(root), "view", "--song", "t"]) == 0
    out = capsys.readouterr().out
    assert "pending" in out      # A-01 还没译
    assert "skip" in out         # A-02 keep_source
    assert "1/2 句已定稿" in out  # keep_source 算已完成


def test_run_without_api_exits_2_and_keeps_request(root, capsys):
    _song(root)
    assert main(["--root", str(root), "run", "--song", "t"]) == 2
    err = capsys.readouterr().err
    assert "request.json" in err
    assert len(list((root / "t" / "runs").glob("*/request.json"))) == 1


def test_api_client_without_key_exits_1(root, capsys, monkeypatch):
    from dubbing import cli

    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: [])
    _song(root)
    assert main(["--root", str(root), "run", "--song", "t", "--client", "api"]) == 1
    err = capsys.readouterr().err
    assert "API key" in err


def test_export_command_writes_lyrics(root, capsys):
    _song(root)
    assert main(["--root", str(root), "export", "--song", "t"]) == 0
    assert (root / "t" / "lyrics.zh.txt").is_file()
    assert (root / "t" / "lyrics.md").is_file()
    out = capsys.readouterr().out
    assert "lyrics.zh.txt" in out


def test_runs_and_show_commands_work_without_products(root, capsys):
    _song(root)
    assert main(["--root", str(root), "runs", "--song", "t"]) == 0
    assert "还没有任何 run" in capsys.readouterr().out

    assert main(["--root", str(root), "show", "--song", "t"]) == 0
    assert "还没有产物" in capsys.readouterr().out


# -- 默认清空 ------------------------------------------------------------


def _scripted(section="S1", text="一二三四"):
    from dubbing.llm import ScriptedClient

    from helpers import analysis_json, draft_json, scripted

    return ScriptedClient(
        {
            "a": scripted(analysis_json([section])),
            "b": scripted(draft_json(section, {"A-01": text})),
        }
    )


def test_default_run_wipes_and_does_not_reuse_anything(root, monkeypatch, capsys):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    assert main(["--root", str(root), "run", "--song", "t", "--client", "api"]) == 0

    runs_dir = root / "t" / "runs"
    assert len(list(runs_dir.iterdir())) == 3  # a + b + c

    second = _scripted()
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: second)
    assert main(["--root", str(root), "run", "--song", "t", "--client", "api"]) == 0

    # A 也必须重跑，不许复用
    assert [c["agent"] for c in second.calls] == ["a", "b"]
    # 旧结果被删干净，只剩这一轮的
    assert len(list(runs_dir.iterdir())) == 3
    assert "已删除旧结果" in capsys.readouterr().err


def test_keep_history_archives_instead_of_deleting(root, monkeypatch):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])
    first_runs = {p.name for p in (root / "t" / "runs").iterdir()}

    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    assert (
        main(
            [
                "--root",
                str(root),
                "run",
                "--song",
                "t",
                "--client",
                "api",
                "--keep-history",
            ]
        )
        == 0
    )

    archives = list((root / "t" / "archive").glob("*"))
    assert len(archives) == 1
    assert {p.name for p in archives[0].iterdir()} == first_runs


def test_only_does_not_wipe_other_sections(root, monkeypatch):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])
    before = {p.name for p in (root / "t" / "runs").iterdir()}

    used = _scripted()
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: used)
    assert (
        main(
            ["--root", str(root), "run", "--song", "t", "--client", "api", "--only", "S1"]
        )
        == 0
    )

    after = {p.name for p in (root / "t" / "runs").iterdir()}
    assert before <= after          # 旧 run 还在
    assert [c["agent"] for c in used.calls] == ["b"]  # A 复用，没重跑


def test_pending_client_never_wipes(root, monkeypatch, capsys):
    from dubbing import cli

    real_build_client = cli.build_client
    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])
    before = {p.name for p in (root / "t" / "runs").iterdir()}
    capsys.readouterr()

    monkeypatch.setattr(cli, "build_client", real_build_client)
    assert main(["--root", str(root), "run", "--song", "t"]) == 2
    assert {p.name for p in (root / "t" / "runs").iterdir()}.issuperset(before)
    assert "pending 客户端不清空" in capsys.readouterr().err
