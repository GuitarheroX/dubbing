"""提示词装载。

A 和 B 共用一个总提示词（``prompts/shared.md``），再各自拼上角色提示词
（``prompts/a.md`` / ``prompts/b.md``）。拼好的整段文本会算一个 hash 记进
``meta.json``，这样"这版译词是哪版提示词产出的"永远查得到。

注意：凡是"该怎么写""别写什么"的要求都只应出现在这几个文件里，
数据文件（``song.yaml`` 与各产物）不承担指令。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

__all__ = ["prompts_dir", "load_prompt", "system_for", "prompt_hash", "ROLES"]

ROLES = ("a", "b")
_REPO_ROOT = Path(__file__).resolve().parent.parent


def prompts_dir() -> Path:
    env = os.environ.get("DUBBING_PROMPTS")
    return Path(env).resolve() if env else _REPO_ROOT / "prompts"


def load_prompt(name: str) -> str:
    path = prompts_dir() / name
    if not path.is_file():
        raise FileNotFoundError(f"找不到提示词 {path}")
    return path.read_text(encoding="utf-8").strip()


def system_for(role: str) -> str:
    """共享总提示词 + 角色提示词。"""
    if role not in ROLES:
        raise KeyError(f"未知角色 {role!r}，可选 {ROLES}")
    return "\n\n".join([load_prompt("shared.md"), load_prompt(f"{role}.md")])


def prompt_hash(system: str) -> str:
    return hashlib.sha256(system.encode("utf-8")).hexdigest()
