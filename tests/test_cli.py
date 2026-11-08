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
    assert "workflow 0 run" in out


def test_dry_run_writes_preview_under_workflow(root, capsys):
    _song(root)
    assert main(["--root", str(root), "run", "--song", "t", "--dry-run"]) == 0

    previews = list((root / "t" / "workflow").glob("preview-*"))
    assert len(previews) == 1
    assert (previews[0] / "perceiver.system.md").is_file()
    assert (previews[0] / "perceiver.data.json").is_file()
    assert (previews[0] / "writer-S1.data.json").is_file()
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
    store = root / "t" / "workflow"
    requests = list(store.glob("run-*/perceiver/request.json"))
    assert len(requests) == 1


def test_api_client_without_key_exits_1(root, capsys, monkeypatch):
    from dubbing import cli

    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: [])
    _song(root)
    assert main(["--root", str(root), "run", "--song", "t", "--client", "api"]) == 1
    err = capsys.readouterr().err
    assert "API key" in err


def test_export_without_a_run_is_an_error(root, capsys):
    _song(root)
    assert main(["--root", str(root), "export", "--song", "t"]) == 1
    assert "没有可导出的歌词" in capsys.readouterr().err


def test_runs_and_show_commands_work_without_products(root, capsys):
    _song(root)
    assert main(["--root", str(root), "runs", "--song", "t"]) == 0
    assert "还没有任何 run" in capsys.readouterr().out

    assert main(["--root", str(root), "show", "--song", "t"]) == 0
    assert "还没有任何 run" in capsys.readouterr().out


# -- 每次 run 一个目录 ----------------------------------------------------


def _scripted(section="S1", text="一二三四"):
    from dubbing.llm import ScriptedClient

    from helpers import analysis_json, draft_json, scripted

    return ScriptedClient(
        {
            "perceiver": scripted(analysis_json([section])),
            "writer": scripted(draft_json(section, {"A-01": text})),
        }
    )


def test_each_run_makes_a_new_directory_and_reuses_analysis(root, monkeypatch, capsys):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    assert main(["--root", str(root), "run", "--song", "t", "--client", "api"]) == 0

    workflow = root / "t" / "workflow"
    first = sorted(p.name for p in workflow.glob("run-*"))
    assert len(first) == 1

    second = _scripted()
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: second)
    assert main(["--root", str(root), "run", "--song", "t", "--client", "api"]) == 0

    # 新 run 新目录，旧的不动
    assert len(sorted(workflow.glob("run-*"))) == 2
    # perceiver 复用，不重复花钱；writer 重跑
    assert [c["agent"] for c in second.calls] == ["writer"]
    assert (workflow / "latest.json").is_file()
    assert len(list(workflow.glob("run-*/lyrics.zh.txt"))) == 2


def test_redo_analysis_forces_a_fresh_a(root, monkeypatch):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])

    second = _scripted()
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: second)
    assert (
        main(["--root", str(root), "run", "--song", "t", "--client", "api", "--redo-analysis"])
        == 0
    )
    assert [c["agent"] for c in second.calls] == ["perceiver", "writer"]


def test_only_carries_other_sections_and_reuses_analysis(root, monkeypatch):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])
    workflow = root / "t" / "workflow"
    before = {p.name for p in workflow.glob("run-*")}

    used = _scripted()
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: used)
    assert (
        main(["--root", str(root), "run", "--song", "t", "--client", "api", "--only", "S1"])
        == 0
    )

    after = {p.name for p in workflow.glob("run-*")}
    assert before <= after          # 旧 run 还在
    assert [c["agent"] for c in used.calls] == ["writer"]  # perceiver 复用，没重跑


def test_pending_client_does_not_touch_previous_runs(root, monkeypatch, capsys):
    from dubbing import cli

    real_build_client = cli.build_client
    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])
    workflow = root / "t" / "workflow"
    before = {p.name for p in workflow.glob("run-*")}
    capsys.readouterr()

    monkeypatch.setattr(cli, "build_client", real_build_client)
    assert main(["--root", str(root), "run", "--song", "t"]) == 2
    after = {p.name for p in workflow.glob("run-*")}
    assert before <= after
    assert len(after) == len(before) + 1  # pending 那次只留下入参
    assert "停在模型调用处" in capsys.readouterr().err


# -- 批量 ----------------------------------------------------------------


def _two_songs(root):
    for sid in ("s1", "s2"):
        write_song(root, sid, ["S1"], [phrase("A-01", "S1", "ああああ", 4)])


def _multi(count, section="S1", text="一二三四"):
    from dubbing.llm import ScriptedClient

    from helpers import analysis_json, draft_json, scripted

    return ScriptedClient(
        {
            "perceiver": scripted(*[analysis_json([section]) for _ in range(count)]),
            "writer": scripted(*[draft_json(section, {"A-01": text}) for _ in range(count)]),
        }
    )


def test_run_needs_song_or_all(root, capsys):
    _two_songs(root)
    assert main(["--root", str(root), "run", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "要么给 --song" in err
    assert "s1" in err


def test_all_runs_every_song(root, monkeypatch):
    from dubbing import cli

    _two_songs(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _multi(2))
    assert main(["--root", str(root), "run", "--all", "--client", "api"]) == 0

    for sid in ("s1", "s2"):
        assert (root / sid / "workflow" / "latest.json").is_file()
        assert len(list((root / sid / "workflow").glob("run-*/lyrics.zh.txt"))) == 1


def test_song_can_be_repeated(root, monkeypatch):
    from dubbing import cli

    _two_songs(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _multi(2))
    assert (
        main(
            ["--root", str(root), "run", "--song", "s1", "--song", "s2", "--client", "api"]
        )
        == 0
    )
    assert (root / "s1" / "workflow" / "latest.json").is_file()
    assert (root / "s2" / "workflow" / "latest.json").is_file()


def test_batch_keeps_going_after_one_song_fails(root, monkeypatch, capsys):
    from dubbing import cli

    _two_songs(root)
    # s1 三次都超字数 → exhausted；s2 一次过
    from dubbing.llm import ScriptedClient

    from helpers import analysis_json, draft_json, scripted

    client = ScriptedClient(
        {
            "perceiver": scripted(analysis_json(["S1"]), analysis_json(["S1"])),
            "writer": scripted(
                draft_json("S1", {"A-01": "一二三四五"}),
                draft_json("S1", {"A-01": "一二三四五"}),
                draft_json("S1", {"A-01": "一二三四五"}),
                draft_json("S1", {"A-01": "一二三四"}),
            ),
        }
    )
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: client)
    assert main(["--root", str(root), "run", "--all", "--client", "api"]) == 1

    out = capsys.readouterr().out
    assert "1 首成功 / 1 首失败" in out
    assert "✗ s1" in out
    assert (root / "s2" / "workflow" / "latest.json").is_file()


def test_prompt_command_prints_system_and_user(root, capsys):
    _song(root)
    assert (
        main(["--root", str(root), "prompt", "--song", "t", "--agent", "writer", "--section", "S1"])
        == 0
    )
    out = capsys.readouterr().out
    assert "# 共同前提" in out          # system = shared.md + writer.md
    assert "本次任务数据（JSON）" in out  # user = 数据
    assert "system" in out and "user" in out


def test_prompt_command_reads_a_real_run(root, capsys, monkeypatch):
    from dubbing import cli

    _song(root)
    monkeypatch.setattr(cli, "build_client", lambda *a, **k: _scripted())
    main(["--root", str(root), "run", "--song", "t", "--client", "api"])

    run_id = sorted(p.name for p in (root / "t" / "workflow").glob("run-*"))[0]
    capsys.readouterr()
    assert main(["--root", str(root), "prompt", "--song", "t", "--run", run_id]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "真实发出去的请求" in out
