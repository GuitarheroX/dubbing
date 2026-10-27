"""可唱性检查里唯一需要的"数数"逻辑：一段中文译词唱出来是几个音节。

规则（写死在这里，不要在别处再实现一遍）：

- 汉字：1 个 = 1 音节。
- 假名：1 个 = 1 音节。译词里出现假名本身是违规（NON_ZH），这里仍计 1，
  这样报错时给 B 的数字才对得上。
- 空白、标点、符号、控制字符：0。
- 连续拉丁字母：按元音组估算，`baby` = 2，`stray` = 1，`rhythm` = 1。
- 阿拉伯数字：逐位记 1（`1999` = 4）。

已知局限：

- 儿化（`花儿`）按 2 计，实际是 1。译配里"儿"很少用，先不处理。
- 外文按元音组估算，多音节词可能差 1。
- 这些局限只影响 C 的报数，不影响数据模型；要改只改本文件。
"""

from __future__ import annotations

import unicodedata

__all__ = [
    "count_syllables",
    "contains_kana",
    "has_punct",
    "is_hanzi",
    "is_kana",
    "is_punct",
    "is_silent",
]

_VOWELS = "aeiouy"


def is_hanzi(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3400 <= o <= 0x4DBF        # 扩展 A
        or 0x4E00 <= o <= 0x9FFF     # 基本区
        or 0xF900 <= o <= 0xFAFF     # 兼容
        or 0x20000 <= o <= 0x2FA1F   # 扩展 B 及以后
    )


def is_kana(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF        # 平假名 + 片假名
        or 0x31F0 <= o <= 0x31FF     # 片假名扩展
        or 0xFF66 <= o <= 0xFF9D     # 半角片假名
    )


def is_punct(ch: str) -> bool:
    """标点（不含空格）。空格是允许的行内停顿，不算标点。"""
    return unicodedata.category(ch).startswith("P")


def is_silent(ch: str) -> bool:
    """标点、空白、符号、控制字符——都唱不出声，计数时一律跳过。"""
    return unicodedata.category(ch)[0] in {"P", "Z", "S", "C"}


def _is_digit(ch: str) -> bool:
    return ch.isascii() and ch.isdigit()


def _latin_token_syllables(tok: str) -> int:
    n = 0
    i = 0
    while i < len(tok):
        ch = tok[i].lower()
        if ch.isdigit():
            n += 1
            i += 1
            continue
        if ch in _VOWELS:
            n += 1
            while i < len(tok) and tok[i].lower() in _VOWELS:
                i += 1
            continue
        i += 1
    return max(n, 1)


def count_syllables(text: str) -> int:
    n = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if is_hanzi(ch) or is_kana(ch):
            n += 1
            i += 1
            continue
        if _is_digit(ch) or (ch.isascii() and ch.isalpha()):
            j = i
            while j < len(text) and (
                _is_digit(text[j]) or (text[j].isascii() and text[j].isalpha())
            ):
                j += 1
            n += _latin_token_syllables(text[i:j])
            i = j
            continue
        i += 1  # 标点、空白、符号
    return n


def contains_kana(text: str) -> bool:
    return any(is_kana(ch) for ch in text)


def has_punct(text: str) -> bool:
    return any(is_punct(ch) for ch in text)
