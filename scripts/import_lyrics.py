#!/usr/bin/env python3
"""从「歌词 总字数 [分句...]」格式的标注文本导入 ``song.yaml``。

用法::

    python scripts/import_lyrics.py data/lyrics.txt
    python scripts/import_lyrics.py data/lyrics.txt --dry-run
    python scripts/import_lyrics.py data/lyrics.txt --force

真正的解析逻辑在 ``dubbing/import_lyrics.py``；这个文件只负责把仓库根目录
放进 ``sys.path``，这样不安装包也能直接跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.import_lyrics import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
