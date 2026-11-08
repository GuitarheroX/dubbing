"""one-shot baseline：workflow 的 writer 提示词 + 单次调用，没有 perceiver 分析、没有 reviewer 复查。"""

import json

from baseline.run_baselines import build_one_shot_request, run_one_shot
from dubbing.llm import ScriptedClient
from dubbing.models import Song
from dubbing.store import RunStore
from dubbing.view import build_view

from helpers import draft_json, phrase, song_json


def _song():
    return Song.from_json(
        song_json(
            "test",
            ["V1", "C1"],
            [
                phrase("V1-01", "V1", "あああ", 6),
                phrase("V1-02", "V1", "らら", keep_source=True),
                phrase("C1-01", "C1", "いい", 4),
            ],
        )
    )


def test_one_shot_input_is_the_workflow_prompt_with_full_context():
    request = build_one_shot_request(_song(), "C1", {"V1-01": "一二三四五六"})
    # 用的是 workflow 的 shared.md + b.md
    assert "# 共同前提" in request.system
    assert "写词的人" in request.system

    rows = {r["id"]: r for r in request.data["phrases"]}
    assert set(rows) == {"V1-01", "V1-02", "C1-01"}      # 整首歌都看得到
    assert rows["V1-01"]["zh"] == "一二三四五六"          # 已经译好的段
    assert rows["V1-02"]["keep_source"] is True
    assert rows["C1-01"]["budget"] == {"min": 4, "max": 4}
    assert request.data["focus"] == "C1"
    # 没有 A 的体会：analysis 是空的，也永远不会有反馈
    assert request.data["analysis"]["overall"]["theme"] == ""
    assert request.data["analysis"]["imagery_pool"] == []


def test_one_shot_runs_section_by_section_and_carries_accepted_texts(tmp_path):
    song = _song()
    client = ScriptedClient(
        {
            "writer": [
                json.dumps(draft_json("V1", {"V1-01": "一二三四五六"}), ensure_ascii=False),
                json.dumps(draft_json("C1", {"C1-01": "一二三四"}), ensure_ascii=False),
            ]
        }
    )
    out = tmp_path / "one-shot"
    run_one_shot(song, client, out=out)

    assert [c["agent"] for c in client.calls] == ["writer", "writer"]
    first = json.loads(client.calls[0]["user"].split("```json")[1].split("```")[0])
    assert first["focus"] == "V1"
    assert "zh" not in {r["id"]: r for r in first["phrases"]}["C1-01"]
    # 第二段能看到第一段已定稿的译词
    second = json.loads(client.calls[1]["user"].split("```json")[1].split("```")[0])
    assert {r["id"]: r for r in second["phrases"]}["V1-01"]["zh"] == "一二三四五六"


def test_one_shot_writes_one_run_with_three_files_per_record(tmp_path):
    song = _song()
    client = ScriptedClient(
        {
            "writer": [
                json.dumps(draft_json("V1", {"V1-01": "一二三四五六"}), ensure_ascii=False),
                json.dumps(draft_json("C1", {"C1-01": "一二三四"}), ensure_ascii=False),
            ]
        }
    )
    out = tmp_path / "one-shot"
    run_one_shot(song, client, out=out)

    store = RunStore(out)
    run_id = store.latest_run()
    assert run_id is not None
    for index in (1, 2):
        names = sorted(p.name for p in store.record_dir(run_id, "writer", index).iterdir())
        assert names == ["meta.json", "output.json", "request.json"]
    run_dir = store.run_dir(run_id)
    assert (run_dir / "lyrics.zh.txt").is_file()
    assert (run_dir / "lyrics.md").is_file()
    # one-shot 不产生 reviewer 记录
    assert not store.has_output(run_id, "reviewer", 1)

    view = build_view(song, store, run_id)
    assert view.row("V1-02").verdict == "skip"
    assert view.row("V1-01").sing_text == "一二三四五六"
    assert view.row("C1-01").sing_text == "一二三四"


def test_one_shot_dry_run_writes_requests_only(tmp_path):
    song = _song()
    client = ScriptedClient([])
    out = tmp_path / "one-shot"
    run_one_shot(song, client, out=out, dry_run=True)

    store = RunStore(out)
    assert client.calls == []  # 一次模型都没调
    previews = list(out.glob("preview-*"))
    assert len(previews) == 1
    assert (previews[0] / "sections" / "001" / "writer" / "request.json").is_file()
    assert not (previews[0] / "sections" / "001" / "writer" / "output.json").exists()
    # 预览不算一次运行
    assert store.list_runs() == []
    assert store.latest_run() is None
