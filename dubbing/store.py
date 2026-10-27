"""磁盘布局：run 目录、``meta.json``、``state.yaml``。

一次 run = 一次 agent 调用 = 一个目录，永不覆盖：

    data/songs/<song_id>/runs/<ts>_<agent>_<input_hash>/
        request.json    调用的完整入参（渲染后的 messages + 结构化数据）
        output.json     产物
        meta.json       血缘：model / prompt_hash / input_hash / ts

``state.yaml`` 只放指针，决定"当前视图由哪几个 run 拼成"。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .models import State

__all__ = ["Request", "Run", "SongStore"]


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass(frozen=True)
class Request:
    """一次调用的入参。``data`` 是结构化数据，``system`` / ``user`` 是渲染后的消息。"""

    system: str
    user: str
    data: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "messages": {"system": self.system, "user": self.user},
            "data": self.data,
        }

    def digest(self) -> str:
        return _hash(self.system, self.user, _canonical(self.data))


@dataclass(frozen=True)
class Run:
    id: str
    dir: Path
    agent: str
    input_hash: str

    @property
    def request_path(self) -> Path:
        return self.dir / "request.json"

    @property
    def output_path(self) -> Path:
        return self.dir / "output.json"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"


class SongStore:
    def __init__(self, root: Path | str, song_id: str) -> None:
        self.root = Path(root)
        self.song_id = song_id
        self.song_dir = self.root / song_id
        self.runs_dir = self.song_dir / "runs"
        self.state_path = self.song_dir / "state.yaml"

    # -- runs ------------------------------------------------------------

    def new_run(self, agent: str, request: Request) -> Run:
        """建目录并写下 ``request.json``。调用前就落盘，便于"API 还没接"时检查入参。"""
        digest = request.digest()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = f"{ts}_{agent}_{digest[:6]}"
        rid = base
        n = 2
        while (self.runs_dir / rid).exists():
            rid = f"{base}-{n}"
            n += 1
        run = Run(id=rid, dir=self.runs_dir / rid, agent=agent, input_hash=digest)
        run.dir.mkdir(parents=True, exist_ok=False)
        self._write(run.request_path, request.to_json())
        return run

    def finish(self, run: Run, output: dict[str, Any], *, meta: dict[str, Any]) -> None:
        self._write(run.output_path, output)
        self._write(
            run.meta_path,
            {
                "agent": run.agent,
                "run_id": run.id,
                "input_hash": run.input_hash,
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **meta,
            },
        )

    def abandon(self, run: Run, *, reason: str) -> None:
        self._write(
            run.meta_path,
            {
                "agent": run.agent,
                "run_id": run.id,
                "input_hash": run.input_hash,
                "abandoned": reason,
                "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
        )

    def has_run(self, run_id: str | None) -> bool:
        return bool(run_id) and (self.runs_dir / str(run_id) / "output.json").is_file()

    def load_output(self, run_id: str) -> dict[str, Any]:
        path = self.runs_dir / run_id / "output.json"
        if not path.is_file():
            raise FileNotFoundError(f"run {run_id} 没有 output.json：{path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_request(self, run_id: str) -> dict[str, Any]:
        path = self.runs_dir / run_id / "request.json"
        if not path.is_file():
            raise FileNotFoundError(f"run {run_id} 没有 request.json：{path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_meta(self, run_id: str) -> dict[str, Any]:
        return json.loads((self.runs_dir / run_id / "meta.json").read_text(encoding="utf-8"))

    def run_ids(self, agent: str | None = None) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        ids = sorted(p.name for p in self.runs_dir.iterdir() if p.is_dir())
        if agent is None:
            return ids
        return [r for r in ids if self._agent_of(r) == agent]

    @staticmethod
    def _agent_of(run_id: str) -> str:
        parts = run_id.split("_")
        return parts[1] if len(parts) > 1 else ""

    # -- 清空 ------------------------------------------------------------

    def reset(self, *, keep_history: bool = False) -> dict[str, object]:
        """删除全部 run 与 state。``keep_history=True`` 时改为移进 ``archive/<ts>/``。

        返回做掉了什么，给命令行报一句。
        """
        run_ids = self.run_ids()
        moved_to: Path | None = None
        if keep_history and run_ids:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            moved_to = self.song_dir / "archive" / stamp
            moved_to.mkdir(parents=True, exist_ok=True)
            for rid in run_ids:
                shutil.move(str(self.runs_dir / rid), str(moved_to / rid))
        else:
            shutil.rmtree(self.runs_dir, ignore_errors=True)

        had_state = self.state_path.is_file()
        self.state_path.unlink(missing_ok=True)
        return {"runs": len(run_ids), "state": had_state, "archive": moved_to}

    # -- state -----------------------------------------------------------

    def load_state(self) -> State:
        if not self.state_path.is_file():
            return State()
        raw = yaml.safe_load(self.state_path.read_text(encoding="utf-8"))
        return State.from_json(raw)

    def save_state(self, state: State) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            state.to_json(), allow_unicode=True, sort_keys=False, default_flow_style=False
        )
        self.state_path.write_text(text, encoding="utf-8")

    # -- 内部 ------------------------------------------------------------

    @staticmethod
    def _write(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
