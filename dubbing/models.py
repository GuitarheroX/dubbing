"""数据结构。

三类东西：

1. L0：``Song`` —— 由人标注，agent 只读。
2. 产物：``Analysis``（A）/ ``Draft``（B）/ ``Review``（C）。
3. 视图与状态：``View`` / ``State``。

所有 ``from_json`` 只做形状与自洽性校验（字段在不在、类型对不对、
区间反没反），跨对象引用（比如 section id 是否存在）交给调用方，
这样加载单个产物文件不需要同时拿到 Song。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "SchemaError",
    "Budget",
    "Section",
    "Phrase",
    "Song",
    "Imagery",
    "Analysis",
    "DraftPhrase",
    "Draft",
    "Violation",
    "Changes",
    "Review",
    "ViewRow",
    "View",
    "SectionState",
    "State",
]


class SchemaError(ValueError):
    """字段缺失、类型不对或自相矛盾。"""


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _need(obj: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in obj:
        raise SchemaError(f"{where}: 缺少字段 {key!r}")
    return obj[key]


def _str(obj: Mapping[str, Any], key: str, where: str, *, default: Any = ...) -> str:
    if key not in obj:
        if default is ...:
            raise SchemaError(f"{where}: 缺少字段 {key!r}")
        return default
    v = obj[key]
    if not isinstance(v, str):
        raise SchemaError(f"{where}.{key}: 期望字符串，得到 {type(v).__name__}")
    return v


def _int(obj: Mapping[str, Any], key: str, where: str) -> int:
    v = _need(obj, key, where)
    if isinstance(v, bool) or not isinstance(v, int):
        raise SchemaError(f"{where}.{key}: 期望整数，得到 {type(v).__name__}")
    return v


def _list(obj: Mapping[str, Any], key: str, where: str) -> list:
    v = _need(obj, key, where)
    if not isinstance(v, list):
        raise SchemaError(f"{where}.{key}: 期望列表，得到 {type(v).__name__}")
    return v


def _map(obj: Mapping[str, Any], key: str, where: str) -> dict:
    v = _need(obj, key, where)
    if not isinstance(v, dict):
        raise SchemaError(f"{where}.{key}: 期望对象，得到 {type(v).__name__}")
    return v


def _require_mapping(obj: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise SchemaError(f"{where}: 期望对象，得到 {type(obj).__name__}")
    return obj


# --------------------------------------------------------------------------
# L0：人工标注
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """可唱汉字数区间。MVP 阶段人工标定值，即 ``min == max``。"""

    min: int
    max: int

    def __post_init__(self) -> None:
        if self.min < 0 or self.max < 0:
            raise SchemaError(f"budget 不能为负：{self.min}..{self.max}")
        if self.min > self.max:
            raise SchemaError(f"budget 区间颠倒：{self.min} > {self.max}")

    def allows(self, n: int) -> bool:
        return self.min <= n <= self.max

    def as_tuple(self) -> tuple[int, int]:
        return (self.min, self.max)

    def to_json(self) -> dict[str, int]:
        return {"min": self.min, "max": self.max}

    @classmethod
    def from_json(cls, obj: Any, where: str = "budget") -> "Budget":
        m = _require_mapping(obj, where)
        return cls(min=_int(m, "min", where), max=_int(m, "max", where))


@dataclass(frozen=True)
class Section:
    id: str
    role: str = ""

    @classmethod
    def from_json(cls, obj: Any, where: str = "section") -> "Section":
        m = _require_mapping(obj, where)
        return cls(id=_str(m, "id", where), role=_str(m, "role", where, default=""))


@dataclass(frozen=True)
class Phrase:
    """一个乐句。歌词的最小单位，也是预算、译文、检查结果的对齐单位。"""

    id: str
    section: str
    ja: str
    budget: Budget | None = None
    keep_source: bool = False
    notes: str = ""
    #: 这一句在旋律上被切成几段、每段几个音，例如 ``(3, 4, 5)``。
    #: ``None`` 表示未标注，等价于"整句一段"（见 ``parts``）。
    pattern: tuple[int, ...] | None = None

    @property
    def translatable(self) -> bool:
        return not self.keep_source

    @property
    def parts(self) -> tuple[int, ...]:
        """归一化后的分段：没标注就是整句一段。"""
        if self.pattern is not None:
            return self.pattern
        if self.budget is not None:
            return (self.budget.max,)
        return ()

    @classmethod
    def from_json(cls, obj: Any, where: str = "phrase") -> "Phrase":
        m = _require_mapping(obj, where)
        pid = _str(m, "id", where)
        where = f"phrase[{pid}]"
        raw_budget = m.get("budget")
        budget = None if raw_budget is None else Budget.from_json(raw_budget, f"{where}.budget")
        keep = m.get("keep_source", False)
        if not isinstance(keep, bool):
            raise SchemaError(f"{where}.keep_source: 期望布尔，得到 {type(keep).__name__}")
        if keep and budget is not None:
            raise SchemaError(f"{where}: keep_source 的乐句不应有 budget（应写 null）")

        raw_pattern = m.get("pattern")
        pattern: tuple[int, ...] | None = None
        if raw_pattern is not None:
            if not isinstance(raw_pattern, (list, tuple)) or not raw_pattern:
                raise SchemaError(f"{where}.pattern: 期望非空数组，例如 [3, 4, 5]")
            parts: list[int] = []
            for i, v in enumerate(raw_pattern):
                if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                    raise SchemaError(f"{where}.pattern[{i}]: 期望正整数，得到 {v!r}")
                parts.append(v)
            pattern = tuple(parts)

        if pattern is not None:
            if keep:
                raise SchemaError(f"{where}: keep_source 的乐句不应有 pattern（整句保留原文）")
            if budget is None:
                raise SchemaError(f"{where}: 写了 pattern 就必须写 budget")
            total = sum(pattern)
            if not budget.allows(total):
                raise SchemaError(
                    f"{where}: pattern 合计 {total} 字，超出 budget "
                    f"{budget.min}-{budget.max}"
                )

        return cls(
            id=pid,
            section=_str(m, "section", where),
            ja=_str(m, "ja", where),
            budget=budget,
            keep_source=keep,
            notes=_str(m, "notes", where, default=""),
            pattern=pattern,
        )

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "section": self.section,
            "ja": self.ja,
            "budget": None if self.budget is None else self.budget.to_json(),
        }
        if self.keep_source:
            d["keep_source"] = True
        if self.notes:
            d["notes"] = self.notes
        if self.pattern is not None:
            d["pattern"] = list(self.pattern)
        return d


@dataclass(frozen=True)
class Song:
    id: str
    title: str
    artist: str
    source_lang: str
    sections: tuple[Section, ...]
    phrases: tuple[Phrase, ...]

    def __post_init__(self) -> None:
        if not self.sections:
            raise SchemaError("song: sections 为空")
        if not self.phrases:
            raise SchemaError("song: phrases 为空")

        sids = [s.id for s in self.sections]
        if len(set(sids)) != len(sids):
            raise SchemaError(f"song: section id 重复 {sids}")
        known = set(sids)

        pids = [p.id for p in self.phrases]
        if len(set(pids)) != len(pids):
            dup = sorted({x for x in pids if pids.count(x) > 1})
            raise SchemaError(f"song: phrase id 重复 {dup}")

        for p in self.phrases:
            if p.section not in known:
                raise SchemaError(f"phrase[{p.id}]: 未知 section {p.section!r}")
        if not any(p.translatable for p in self.phrases):
            raise SchemaError("song: 没有任何可译乐句")

    # -- 查询 ------------------------------------------------------------

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.sections)

    def section(self, sid: str) -> Section:
        for s in self.sections:
            if s.id == sid:
                return s
        raise KeyError(f"未知 section {sid!r}")

    def phrases_of(self, sid: str) -> tuple[Phrase, ...]:
        return tuple(p for p in self.phrases if p.section == sid)

    def translatable_of(self, sid: str) -> tuple[Phrase, ...]:
        return tuple(p for p in self.phrases_of(sid) if p.translatable)

    def phrase(self, pid: str) -> Phrase:
        for p in self.phrases:
            if p.id == pid:
                return p
        raise KeyError(f"未知 phrase {pid!r}")

    @property
    def phrase_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.phrases)

    @classmethod
    def from_json(cls, obj: Any, where: str = "song") -> "Song":
        m = _require_mapping(obj, where)
        meta = _require_mapping(_need(m, "song", where), f"{where}.song")
        return cls(
            id=_str(meta, "id", f"{where}.song"),
            title=_str(meta, "title", f"{where}.song"),
            artist=_str(meta, "artist", f"{where}.song", default=""),
            source_lang=_str(meta, "source_lang", f"{where}.song", default="ja"),
            sections=tuple(
                Section.from_json(s, f"{where}.sections[{i}]")
                for i, s in enumerate(_list(m, "sections", where))
            ),
            phrases=tuple(
                Phrase.from_json(p, f"{where}.phrases[{i}]")
                for i, p in enumerate(_list(m, "phrases", where))
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "song": {
                "id": self.id,
                "title": self.title,
                "artist": self.artist,
                "source_lang": self.source_lang,
            },
            "sections": [{"id": s.id, "role": s.role} for s in self.sections],
            "phrases": [p.to_json() for p in self.phrases],
        }


# --------------------------------------------------------------------------
# A 的产物
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Imagery:
    image: str
    why: str = ""

    @classmethod
    def from_json(cls, obj: Any, where: str) -> "Imagery":
        m = _require_mapping(obj, where)
        return cls(image=_str(m, "image", where), why=_str(m, "why", where, default=""))

    def to_json(self) -> dict[str, str]:
        return {"image": self.image, "why": self.why}


@dataclass(frozen=True)
class Analysis:
    """A 的输出：全曲层的意境情感 + 意象池 + 分段情绪。全是描述，无指令。"""

    theme: str
    emotional_arc: str
    narrative: str
    tone_words: tuple[str, ...] = ()
    imagery_pool: tuple[Imagery, ...] = ()
    sections: Mapping[str, str] = field(default_factory=dict)

    def mood_of(self, sid: str) -> str:
        return self.sections.get(sid, "")

    @classmethod
    def from_json(cls, obj: Any, where: str = "analysis") -> "Analysis":
        m = _require_mapping(obj, where)
        overall = _require_mapping(_need(m, "overall", where), f"{where}.overall")
        words = _list(overall, "tone_words", f"{where}.overall") if "tone_words" in overall else []
        for i, w in enumerate(words):
            if not isinstance(w, str):
                raise SchemaError(f"{where}.overall.tone_words[{i}]: 期望字符串")
        sections_raw = _map(m, "sections", where)
        secs: dict[str, str] = {}
        for sid, mood in sections_raw.items():
            if not isinstance(mood, str):
                raise SchemaError(f"{where}.sections[{sid!r}]: 期望字符串")
            secs[str(sid)] = mood
        return cls(
            theme=_str(overall, "theme", f"{where}.overall"),
            emotional_arc=_str(overall, "emotional_arc", f"{where}.overall"),
            narrative=_str(overall, "narrative", f"{where}.overall"),
            tone_words=tuple(words),
            imagery_pool=tuple(
                Imagery.from_json(x, f"{where}.imagery_pool[{i}]")
                for i, x in enumerate(_list(m, "imagery_pool", where))
            ),
            sections=secs,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "overall": {
                "theme": self.theme,
                "emotional_arc": self.emotional_arc,
                "narrative": self.narrative,
                "tone_words": list(self.tone_words),
            },
            "imagery_pool": [i.to_json() for i in self.imagery_pool],
            "sections": dict(self.sections),
        }


# --------------------------------------------------------------------------
# B 的产物
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftPhrase:
    id: str
    zh: str
    chars: int | None = None
    self_check: Mapping[str, Any] | None = None

    @classmethod
    def from_json(cls, obj: Any, where: str) -> "DraftPhrase":
        m = _require_mapping(obj, where)
        pid = _str(m, "id", where)
        where = f"{where}[{pid}]"
        chars = None
        if m.get("chars") is not None:
            chars = _int(m, "chars", where)
        sc = m.get("self_check")
        if sc is not None:
            sc = _require_mapping(sc, f"{where}.self_check")
        return cls(id=pid, zh=_str(m, "zh", where), chars=chars, self_check=sc)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "zh": self.zh}
        if self.chars is not None:
            d["chars"] = self.chars
        if self.self_check is not None:
            d["self_check"] = dict(self.self_check)
        return d


@dataclass(frozen=True)
class Draft:
    """B 的输出：一个 section 的全部译文，按 phrase id 对齐。"""

    section: str
    phrases: tuple[DraftPhrase, ...]

    def by_id(self) -> dict[str, DraftPhrase]:
        return {p.id: p for p in self.phrases}

    def texts(self) -> dict[str, str]:
        return {p.id: p.zh for p in self.phrases}

    @classmethod
    def from_json(cls, obj: Any, where: str = "draft") -> "Draft":
        m = _require_mapping(obj, where)
        return cls(
            section=_str(m, "section", where),
            phrases=tuple(
                DraftPhrase.from_json(p, f"{where}.phrases[{i}]")
                for i, p in enumerate(_list(m, "phrases", where))
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "phrases": [p.to_json() for p in self.phrases],
        }


# --------------------------------------------------------------------------
# C 的产物
# --------------------------------------------------------------------------

#: ``Review.violations[].code`` 的取值，封闭枚举。
CODES = {
    "OVER_BUDGET": "汉字数超出预算上限",
    "UNDER_BUDGET": "汉字数低于预算下限",
    "EMPTY": "没有给出译词",
    "NON_ZH": "译词里残留日文假名",
    "DUPLICATE": "原文不同的乐句给了完全相同的译词",
    "PUNCT": "译词里带标点（唱不出声，且不算字数）",
    "CHAR_MISMATCH": "自报字数与实际字数不一致",
}

HARD_CODES = frozenset({"OVER_BUDGET", "UNDER_BUDGET", "EMPTY", "NON_ZH", "DUPLICATE"})
SOFT_CODES = frozenset({"PUNCT", "CHAR_MISMATCH"})


@dataclass(frozen=True)
class Violation:
    phrase_id: str
    code: str
    severity: str  # "hard" | "soft"
    expected: tuple[int, int] | None = None
    actual: int | None = None
    fix: str = ""

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "phrase_id": self.phrase_id,
            "code": self.code,
            "severity": self.severity,
        }
        if self.expected is not None:
            d["expected"] = list(self.expected)
        if self.actual is not None:
            d["actual"] = self.actual
        if self.fix:
            d["fix"] = self.fix
        return d


@dataclass(frozen=True)
class Changes:
    """本轮相对上一版的改动。只供人看，不算违规。"""

    rewritten: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()

    def to_json(self) -> dict[str, list[str]]:
        return {"rewritten": list(self.rewritten), "unchanged": list(self.unchanged)}


@dataclass(frozen=True)
class Review:
    section: str
    draft_run: str
    verdict: str  # "pass" | "fail"
    violations: tuple[Violation, ...] = ()
    changes: Changes = field(default_factory=Changes)
    regressions: tuple[str, ...] = ()

    @property
    def hard(self) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.severity == "hard")

    def failing_ids(self) -> set[str]:
        return {v.phrase_id for v in self.hard}

    def codes_of(self, phrase_id: str) -> set[str]:
        return {v.code for v in self.violations if v.phrase_id == phrase_id}

    @classmethod
    def from_json(cls, obj: Any, where: str = "review") -> "Review":
        m = _require_mapping(obj, where)
        changes_raw = _require_mapping(_need(m, "changes", where), f"{where}.changes")

        def ids(key: str, at: str) -> tuple[str, ...]:
            out = []
            for i, v in enumerate(_list(changes_raw, key, at)):
                if not isinstance(v, str):
                    raise SchemaError(f"{at}.{key}[{i}]: 期望字符串")
                out.append(v)
            return tuple(out)

        violations = []
        for i, raw in enumerate(_list(m, "violations", where)):
            at = f"{where}.violations[{i}]"
            vm = _require_mapping(raw, at)
            code = _str(vm, "code", at)
            if code not in CODES:
                raise SchemaError(f"{at}.code: 未知错误码 {code!r}")
            exp = vm.get("expected")
            expected = None
            if exp is not None:
                if not isinstance(exp, (list, tuple)) or len(exp) != 2:
                    raise SchemaError(f"{at}.expected: 期望 [min, max]")
                expected = (int(exp[0]), int(exp[1]))
            violations.append(
                Violation(
                    phrase_id=_str(vm, "phrase_id", at),
                    code=code,
                    severity=_str(vm, "severity", at),
                    expected=expected,
                    actual=None if vm.get("actual") is None else int(vm["actual"]),
                    fix=_str(vm, "fix", at, default=""),
                )
            )

        regs = _list(m, "regressions", where)
        return cls(
            section=_str(m, "section", where),
            draft_run=_str(m, "draft_run", where),
            verdict=_str(m, "verdict", where),
            violations=tuple(violations),
            changes=Changes(
                rewritten=ids("rewritten", f"{where}.changes"),
                unchanged=ids("unchanged", f"{where}.changes"),
            ),
            regressions=tuple(str(x) for x in regs),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "draft_run": self.draft_run,
            "verdict": self.verdict,
            "violations": [v.to_json() for v in self.violations],
            "changes": self.changes.to_json(),
            "regressions": list(self.regressions),
        }


# --------------------------------------------------------------------------
# 视图
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ViewRow:
    """装配后的一行乐句：唱出来是什么、什么状态。"""

    phrase_id: str
    section: str
    ja: str
    budget: Budget | None
    lang: str | None  # "zh" | "ja" | None（还没译）
    sing_text: str
    verdict: str  # "pass" | "fail" | "skip" | "pending"
    draft_run: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.phrase_id,
            "section": self.section,
            "ja": self.ja,
            "budget": None if self.budget is None else self.budget.to_json(),
            "lang": self.lang,
            "sing_text": self.sing_text,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class View:
    song_id: str
    rows: tuple[ViewRow, ...]

    def row(self, pid: str) -> ViewRow:
        for r in self.rows:
            if r.phrase_id == pid:
                return r
        raise KeyError(pid)

    def of_section(self, sid: str) -> tuple[ViewRow, ...]:
        return tuple(r for r in self.rows if r.section == sid)

    def accepted_texts(self, *, exclude: Iterable[str] = ()) -> dict[str, str]:
        skip = set(exclude)
        return {
            r.phrase_id: r.sing_text
            for r in self.rows
            if r.lang == "zh" and r.verdict == "pass" and r.phrase_id not in skip
        }

    def to_json(self) -> dict[str, Any]:
        return {"song_id": self.song_id, "rows": [r.to_json() for r in self.rows]}


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------


@dataclass
class SectionState:
    draft: str | None = None
    review: str | None = None
    round: int = 0
    verdict: str | None = None  # "pass" | "fail" | "exhausted" | "skipped"

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"round": self.round}
        for k in ("draft", "review", "verdict"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_json(cls, obj: Any) -> "SectionState":
        m = _require_mapping(obj, "section_state")
        return cls(
            draft=m.get("draft"),
            review=m.get("review"),
            round=int(m.get("round", 0)),
            verdict=m.get("verdict"),
        )


@dataclass
class State:
    analysis: str | None = None
    sections: dict[str, SectionState] = field(default_factory=dict)

    def of(self, sid: str) -> SectionState:
        return self.sections.setdefault(sid, SectionState())

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"sections": {k: v.to_json() for k, v in self.sections.items()}}
        if self.analysis is not None:
            out["song_level"] = {"analysis": self.analysis}
        return out

    @classmethod
    def from_json(cls, obj: Any) -> "State":
        if obj is None:
            return cls()
        m = _require_mapping(obj, "state")
        song_level = m.get("song_level") or {}
        sections = m.get("sections") or {}
        return cls(
            analysis=song_level.get("analysis"),
            sections={
                str(k): SectionState.from_json(v) for k, v in _require_mapping(sections, "state.sections").items()
            },
        )
