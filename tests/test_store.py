"""run 目录存储：一次运行一个目录，每个 agent 只有三个文件。"""

import json

from dubbing.store import Request, RunStore, section_index

from helpers import make_song, phrase


def _seed(store: RunStore, agent: str = "B", index: int | None = 1, text: str = "译词") -> str:
    run_id = store.new_run()
    request = Request(system="s", user="u", data={})
    store.prepare(run_id, agent, request, section_index_=index)
    store.finish(run_id, agent, {"ok": True}, meta={"agent": agent}, section_index_=index)
    return run_id


def test_record_dir_layout(root):
    store = RunStore(root / "t" / "workflow")
    run_id = store.new_run()
    assert store.record_dir(run_id, "A").name == "A"
    assert store.record_dir(run_id, "B", 3) == store.run_dir(run_id) / "sections" / "003" / "B"


def test_run_directory_holds_exactly_three_files_per_agent(root):
    store = RunStore(root / "t" / "workflow")
    run_id = _seed(store)
    (store.run_dir(run_id) / "lyrics.zh.txt").write_text("词\n", encoding="utf-8")
    (store.run_dir(run_id) / "lyrics.md").write_text("# 词\n", encoding="utf-8")
    names = sorted(p.name for p in store.record_dir(run_id, "B", 1).iterdir())
    assert names == ["meta.json", "output.json", "request.json"]


def test_latest_points_at_the_newest_run_and_falls_back(root):
    store = RunStore(root / "t" / "workflow")
    first = _seed(store)
    assert store.latest_run() == first

    store.set_latest(first, status="completed")
    assert store.read_latest()["status"] == "completed"
    assert store.latest_run() == first

    # latest.json 指向一个已经不存在的 run：退回磁盘上最新的一个
    store.set_latest("run-does-not-exist")
    assert store.latest_run() == first


def test_read_back_and_load_output(root):
    store = RunStore(root / "t" / "workflow")
    run_id = _seed(store)
    assert store.has_output(run_id, "B", 1)
    assert store.read_output(run_id, "B", 1) == {"ok": True}
    assert store.read_request(run_id, "B", 1)["messages"]["system"] == "s"
    assert store.read_meta(run_id, "B", 1)["agent"] == "B"
    assert store.load_output(run_id, "C", 1) is None
    assert store.load_output(None, "B", 1) is None


def test_abandon_keeps_reason_and_raw_response(root):
    store = RunStore(root / "t" / "workflow")
    run_id = store.new_run()
    store.abandon(run_id, "A", reason="解析失败：坏了", raw_response="{bad")
    meta = store.read_meta(run_id, "A")
    assert meta["abandoned"] == "解析失败：坏了"
    assert meta["raw_response"] == "{bad"
    assert not store.has_output(run_id, "A")


def test_copy_record_carries_a_section_into_a_new_run(root):
    store = RunStore(root / "t" / "workflow")
    src = _seed(store, "B", 1)
    dst = store.new_run()
    assert store.copy_record(src, dst, "B", section_index_=1)
    assert store.read_output(dst, "B", 1) == {"ok": True}
    assert not store.copy_record(src, dst, "C", section_index_=1)
    assert store.section_indexes(dst) == [1]


def test_section_index_follows_song_yaml_order():
    song = make_song(
        "t", ["S1", "S2"], [phrase("A-01", "S1", "あ", 1), phrase("B-01", "S2", "い", 1)]
    )
    assert section_index(song, "S1") == 1
    assert section_index(song, "S2") == 2


def test_preview_directories_do_not_show_up_as_runs(root):
    store = RunStore(root / "t" / "workflow")
    preview = store.new_preview()
    assert preview.startswith("preview-")
    assert (store.base / preview).is_dir()
    assert store.list_runs() == []
    assert store.latest_run() is None

    run_id = _seed(store)
    assert store.list_runs() == [run_id]


def test_request_json_has_messages_and_data():
    payload = Request(system="s", user="u", data={"x": 1}).to_json()
    assert payload == {"messages": {"system": "s", "user": "u"}, "data": {"x": 1}}
    json.dumps(payload)
