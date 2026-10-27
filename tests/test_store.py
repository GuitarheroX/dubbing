"""清空与归档。"""

import json

from dubbing.store import Request, SongStore


def _seed(store: SongStore, agent: str = "b") -> str:
    run = store.new_run(agent, Request(system="s", user="u", data={}))
    store.finish(run, {"ok": True}, meta={"agent": agent})
    return run.id


def test_reset_deletes_runs_and_state(root):
    store = SongStore(root, "t")
    _seed(store)
    _seed(store, "c")
    store.state_path.write_text("sections: {}\n", encoding="utf-8")

    info = store.reset()

    assert info == {"runs": 2, "state": True, "archive": None}
    assert store.run_ids() == []
    assert not store.state_path.exists()


def test_reset_can_archive_instead_of_delete(root):
    store = SongStore(root, "t")
    first = _seed(store)
    store.state_path.write_text("sections: {}\n", encoding="utf-8")

    info = store.reset(keep_history=True)

    archive = info["archive"]
    assert archive is not None
    assert (archive / first / "output.json").is_file()
    assert store.run_ids() == []
    assert not store.state_path.exists()


def test_reset_on_empty_song_is_harmless(root):
    store = SongStore(root, "t")
    assert store.reset()["runs"] == 0
    assert store.reset(keep_history=True)["archive"] is None


def test_archives_do_not_pollute_run_listing(root):
    store = SongStore(root, "t")
    _seed(store)
    store.reset(keep_history=True)
    assert store.run_ids() == []
    # 新的 run 正常出现
    fresh = _seed(store)
    assert store.run_ids() == [fresh]
