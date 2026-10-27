"""A / B 的入参构造与产物解析。

两条刻意的边界：

- A 的入参**不含 budget**，也不含 `keep_source`。A 一旦看到字数就会开始
  数字数、开始翻译，而它的职责是感受意境。
- B 的入参不含任何指令。指令全在提示词里；``data`` 只说事实。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from .models import CODES, Analysis, Draft, DraftPhrase, SchemaError, Song, Violation
from .prompting import prompt_hash, system_for
from .store import Request

__all__ = [
    "feedback_block",
    "build_a_request",
    "parse_a_output",
    "build_b_request",
    "parse_b_output",
    "b_payload",
]


def _user_message(data: dict[str, Any]) -> str:
    body = json.dumps(data, ensure_ascii=False, indent=2)
    return f"本次任务数据（JSON）：\n```json\n{body}\n```"


def feedback_block(violations: Iterable[Violation]) -> str:
    """把上一轮 C 的结论渲染成给 B 看的返修说明。

    只进消息文本，不进 ``data``：数据文件仍然只放事实，指令留在提示词/消息里。
    """
    items = list(violations)
    if not items:
        return ""
    lines = [
        "上一轮你这一段没过检查。逐句改掉下面的问题（没列出的句子可以保持，"
        "也可以为了让句子更顺而调整）：",
        "",
    ]
    for v in items:
        what = v.fix or CODES.get(v.code, v.code)
        tag = "必修" if v.severity == "hard" else "建议"
        lines.append(f"- [{tag}] {v.phrase_id}：{what}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------


def a_payload(song: Song) -> dict[str, Any]:
    return {
        "song": {
            "title": song.title,
            "artist": song.artist,
            "source_lang": song.source_lang,
        },
        "sections": [
            {
                "id": sec.id,
                "role": sec.role,
                "lyrics": [p.ja for p in song.phrases_of(sec.id)],
            }
            for sec in song.sections
        ],
    }


def build_a_request(song: Song) -> Request:
    system = system_for("a")
    data = a_payload(song)
    return Request(system=system, user=_user_message(data), data=data)


def parse_a_output(obj: Any, song: Song) -> tuple[Analysis, list[str]]:
    analysis = Analysis.from_json(obj)
    warnings: list[str] = []

    known = set(song.section_ids)
    unknown = sorted(set(analysis.sections) - known)
    if unknown:
        warnings.append(f"A 给出了不存在的 section：{unknown}，已丢弃")
    if unknown:
        analysis = Analysis(
            theme=analysis.theme,
            emotional_arc=analysis.emotional_arc,
            narrative=analysis.narrative,
            tone_words=analysis.tone_words,
            imagery_pool=analysis.imagery_pool,
            sections={k: v for k, v in analysis.sections.items() if k in known},
        )

    missing = [sid for sid in song.section_ids if sid not in analysis.sections]
    if missing:
        warnings.append(f"A 没有覆盖 section：{missing}")
    if not analysis.imagery_pool:
        warnings.append("A 的 imagery_pool 为空")
    return analysis, warnings


# --------------------------------------------------------------------------
# B
# --------------------------------------------------------------------------

UNSET: Any = object()


def b_payload(
    song: Song,
    analysis: Analysis,
    *,
    focus: str,
    accepted: dict[str, str] | None = None,
    prev_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """B 的入参。

    - 全曲每个乐句都给（B 需要看到后面唱什么，才不会前后撞意象）。
    - ``accepted``：其他段已定稿的译文，只读参考。
    - ``prev_texts``：**本段**上一版译文，可以改。
    """
    accepted = accepted or {}
    prev_texts = prev_texts or {}
    if focus not in {s.id for s in song.sections}:
        raise KeyError(f"未知 section {focus!r}")

    phrases: list[dict[str, Any]] = []
    for p in song.phrases:
        row: dict[str, Any] = {
            "id": p.id,
            "section": p.section,
            "ja": p.ja,
            "budget": None if p.budget is None else p.budget.to_json(),
        }
        if p.keep_source:
            row["keep_source"] = True
        elif p.section == focus:
            if p.id in prev_texts:
                row["prev_zh"] = prev_texts[p.id]
        elif p.id in accepted:
            row["zh"] = accepted[p.id]
        phrases.append(row)

    return {
        "song": {
            "title": song.title,
            "artist": song.artist,
            "source_lang": song.source_lang,
        },
        "sections": [{"id": s.id, "role": s.role} for s in song.sections],
        "phrases": phrases,
        "focus": focus,
        "analysis": analysis.to_json(),
    }


def build_b_request(
    song: Song,
    analysis: Analysis,
    *,
    focus: str,
    accepted: dict[str, str] | None = None,
    prev_texts: dict[str, str] | None = None,
    feedback: Iterable[Violation] = (),
) -> Request:
    system = system_for("b")
    data = b_payload(
        song, analysis, focus=focus, accepted=accepted, prev_texts=prev_texts
    )
    blocks = [feedback_block(feedback), _user_message(data)]
    user = "\n\n".join(b for b in blocks if b)
    return Request(system=system, user=user, data=data)


def parse_b_output(obj: Any, song: Song, section: str) -> tuple[Draft, list[str]]:
    """宽松解析：能修的都修，修不了留给 C 报 EMPTY。

    模型输出格式出问题是常态，把它当"非法输入"直接抛错会让整轮白跑；
    不如丢掉多余的、补上空缺的，交给 C 用同一套违规机制处理。
    """
    try:
        draft = Draft.from_json(obj)
    except SchemaError as exc:
        raise SchemaError(f"B 的输出结构不对：{exc}") from exc

    warnings: list[str] = []
    if draft.section != section:
        warnings.append(
            f"B 自报 section={draft.section!r}，与请求的 {section!r} 不符，以请求为准"
        )

    translatable = {p.id: p for p in song.translatable_of(section)}
    kept = {p.id for p in song.phrases_of(section) if p.keep_source}
    seen: dict[str, DraftPhrase] = {}

    for dp in draft.phrases:
        pid = dp.id
        if pid in kept:
            warnings.append(f"{pid}: 该句保留原文，B 不应产出译词，已忽略")
            continue
        if pid not in translatable:
            if song.phrase_ids and pid in song.phrase_ids:
                warnings.append(f"{pid}: 属于其他 section，已忽略")
            else:
                warnings.append(f"{pid}: 未知乐句 id，已忽略")
            continue
        if pid in seen:
            warnings.append(f"{pid}: 重复出现，只保留第一次")
            continue
        seen[pid] = dp

    missing = [pid for pid in translatable if pid not in seen]
    for pid in missing:
        warnings.append(f"{pid}: 缺少译词，按空句处理")
        seen[pid] = DraftPhrase(id=pid, zh="")

    ordered = tuple(seen[pid] for pid in translatable if pid in seen)
    return Draft(section=section, phrases=ordered), warnings
