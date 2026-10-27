"""L0 的读写：``data/songs/<song_id>/song.yaml``。"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import SchemaError, Song

__all__ = ["load_song", "song_path", "list_songs"]


def song_path(song_id: str, *, root: Path | str | None = None) -> Path:
    root = Path(root) if root else Path("data/songs")
    return root / song_id / "song.yaml"


def load_song(song_id: str, *, root: Path | str | None = None) -> Song:
    path = song_path(song_id, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    try:
        return Song.from_json(raw, where=str(path))
    except SchemaError as exc:
        raise SchemaError(f"{path}: {exc}") from exc


def list_songs(*, root: Path | str | None = None) -> list[str]:
    root = Path(root) if root else Path("data/songs")
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/song.yaml"))
