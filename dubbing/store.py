"""磁盘布局：**一次运行 = 一个 run 目录**，目录里每个 agent 只有三个文件。

workflow / workflow-RAG 与各 baseline 共用同一套布局：

    <base>/                                  # data/songs/<id>/workflow（或 workflow_rag）
      latest.json                            # 只记“当前 run 是哪个”
      run-<UTC 时间戳>-<8 位随机>/
        perceiver/{request,output,meta}.json # 整曲层的 agent；只有 workflow 系的 perceiver 在这里
        sections/001/writer/{request,output,meta}.json
        sections/001/mimic_lookup/{request,output,meta}.json   # 只有 rag 管线有
        sections/001/mimic_write/{request,output,meta}.json    # 只有 rag 管线有
        sections/001/reviewer/{request,output,meta}.json
        sections/002/...
        lyrics.zh.txt
        lyrics.md

- ``request.json``：这次调用的完整入参，调用前就落盘，方便“API 还没接”时先检查。
- ``output.json``：解析后的产物（Analysis / Draft / LookupResult / Review / lines）。
- ``meta.json``：模型、用量、提示词 hash、轮次等血缘信息；模型原始正文也存在这里
  的 ``raw_response`` 字段，所以解析失败也留得住证据，不必再多一个 ``response.txt``。

一个 run 目录自带全部产物，``latest.json`` 不参与拼装视图，只回答“现在看哪一次”。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import Song

__all__ = [
    "Request",
    "RunStore",
    "RECORD_FILES",
    "section_index",
    "run_base",
]

#: 每个 agent 目录固定只有这三个文件。
RECORD_FILES = ("request.json", "output.json", "meta.json")
RUN_PREFIX = "run-"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def section_index(song: Song, section_id: str) -> int:
    """段落在 ``song.yaml`` 里的 1 起序号，决定 ``sections/001`` 这个名字。"""
    ids = song.section_ids
    if section_id not in ids:
        raise KeyError(f"未知 section {section_id!r}，可选 {list(ids)}")
    return ids.index(section_id) + 1


def run_base(root: Path | str, song_id: str, name: str = "workflow") -> Path:
    """``name`` 取 ``workflow`` 或 ``baselines/<baseline 名>``。"""
    return Path(root) / song_id / name


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


class RunStore:
    """管理 ``<base>/`` 下的若干 run 目录。``base`` 由调用方给定。

    workflow 用 ``data/songs/<id>/workflow``；baseline 用
    ``data/songs/<id>/baselines/<名字>``。两者结构完全一样。
    """

    def __init__(self, base: Path | str) -> None:
        self.base = Path(base)
        self.latest_path = self.base / "latest.json"

    # -- run 目录 --------------------------------------------------------

    def new_run(self) -> str:
        """建一个新 run 目录，返回 run id。时间戳保证排序即先后。"""
        return self._new_dir(RUN_PREFIX)

    def new_preview(self) -> str:
        """``--dry-run`` 用的目录。不叫 ``run-``，所以 ``list_runs`` / ``latest_run`` 看不见它。"""
        return self._new_dir("preview-")

    def _new_dir(self, prefix: str) -> str:
        self.base.mkdir(parents=True, exist_ok=True)
        rid = f"{prefix}{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid4().hex[:8]}"
        (self.base / rid).mkdir(parents=True, exist_ok=False)
        return rid

    def run_dir(self, run_id: str) -> Path:
        return self.base / run_id

    def list_runs(self) -> list[str]:
        if not self.base.is_dir():
            return []
        return sorted(
            p.name for p in self.base.iterdir() if p.is_dir() and p.name.startswith(RUN_PREFIX)
        )

    def latest_run(self) -> str | None:
        """``latest.json`` 指向的 run；文件缺失或指向已删除的目录时退回最新的一个。"""
        if self.latest_path.is_file():
            data = self.read_latest()
            rid = data.get("run")
            if isinstance(rid, str) and (self.base / rid).is_dir():
                return rid
        runs = self.list_runs()
        return runs[-1] if runs else None

    def read_latest(self) -> dict[str, Any]:
        if not self.latest_path.is_file():
            return {}
        try:
            data = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def set_latest(self, run_id: str, **extra: Any) -> None:
        payload: dict[str, Any] = {
            "run": run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        payload.update(extra)
        self._write(self.latest_path, payload)

    # -- 单个 agent 的三个文件 -------------------------------------------

    def record_dir(self, run_id: str, agent: str, section_index_: int | None = None) -> Path:
        """``section_index_`` 为 None 表示整曲层的 agent（如 perceiver）。"""
        root = self.run_dir(run_id)
        if section_index_ is None:
            return root / agent
        return root / "sections" / f"{section_index_:03d}" / agent

    def prepare(
        self, run_id: str, agent: str, request: Request, *, section_index_: int | None = None
    ) -> Path:
        """写下 ``request.json``。调用前落盘。"""
        directory = self.record_dir(run_id, agent, section_index_)
        directory.mkdir(parents=True, exist_ok=True)
        self._write(directory / "request.json", request.to_json())
        return directory

    def finish(
        self,
        run_id: str,
        agent: str,
        output: dict[str, Any],
        *,
        meta: dict[str, Any] | None = None,
        raw_response: str | None = None,
        section_index_: int | None = None,
    ) -> Path:
        directory = self.record_dir(run_id, agent, section_index_)
        directory.mkdir(parents=True, exist_ok=True)
        self._write(directory / "output.json", output)
        self._write(directory / "meta.json", self._meta(agent, meta, raw_response))
        return directory

    def abandon(
        self,
        run_id: str,
        agent: str,
        *,
        reason: str,
        meta: dict[str, Any] | None = None,
        raw_response: str | None = None,
        section_index_: int | None = None,
    ) -> Path:
        directory = self.record_dir(run_id, agent, section_index_)
        directory.mkdir(parents=True, exist_ok=True)
        payload = self._meta(agent, meta, raw_response)
        payload["abandoned"] = reason
        self._write(directory / "meta.json", payload)
        return directory

    @staticmethod
    def _meta(agent: str, meta: dict[str, Any] | None, raw_response: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"agent": agent}
        if raw_response is not None:
            payload["raw_response"] = raw_response
        payload["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        payload.update(meta or {})
        return payload

    # -- 读 --------------------------------------------------------------

    def has_output(self, run_id: str | None, agent: str, section_index_: int | None = None) -> bool:
        if not run_id:
            return False
        return (self.record_dir(run_id, agent, section_index_) / "output.json").is_file()

    def read_request(self, run_id: str, agent: str, section_index_: int | None = None) -> dict[str, Any]:
        path = self.record_dir(run_id, agent, section_index_) / "request.json"
        if not path.is_file():
            raise FileNotFoundError(f"{run_id}/{agent} 没有 request.json：{path}")
        return self._read(path)

    def read_output(self, run_id: str, agent: str, section_index_: int | None = None) -> dict[str, Any]:
        path = self.record_dir(run_id, agent, section_index_) / "output.json"
        if not path.is_file():
            raise FileNotFoundError(f"{run_id}/{agent} 没有 output.json：{path}")
        return self._read(path)

    def read_meta(self, run_id: str, agent: str, section_index_: int | None = None) -> dict[str, Any]:
        path = self.record_dir(run_id, agent, section_index_) / "meta.json"
        if not path.is_file():
            return {}
        return self._read(path)

    def load_output(
        self, run_id: str | None, agent: str, section_index_: int | None = None
    ) -> dict[str, Any] | None:
        """没有 output.json 就返回 None，方便“这一段的这一轮还没跑”。"""
        if not self.has_output(run_id, agent, section_index_):
            return None
        return self.read_output(run_id, agent, section_index_)  # type: ignore[arg-type]

    # -- 复制 ------------------------------------------------------------

    def copy_record(
        self,
        src_run: str,
        dst_run: str,
        agent: str,
        *,
        section_index_: int | None = None,
    ) -> bool:
        """把一个 agent 目录整份抄进另一个 run（``--skip-done`` / ``--only`` 用）。"""
        src = self.record_dir(src_run, agent, section_index_)
        if not src.is_dir():
            return False
        dst = self.record_dir(dst_run, agent, section_index_)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True

    def section_indexes(self, run_id: str) -> list[int]:
        """run 里出现过的段号，升序。"""
        root = self.run_dir(run_id) / "sections"
        if not root.is_dir():
            return []
        return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())

    # -- 内部 ------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} 不是 JSON 对象")
        return data

    @staticmethod
    def _write(path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
