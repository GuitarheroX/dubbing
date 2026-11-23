"""共同计数口径、覆盖率及逐歌配对比较。"""

from collections import Counter
from itertools import combinations
import random

from dubbing.checks import review
from dubbing.counting import count_syllables
from dubbing.models import Draft, DraftPhrase, Song

from .common import VARIANTS, average, ratio, variants_of

METRICS = ("budget_pass_rate", "mean_budget_distance", "under_budget_rate", "over_budget_rate", "rule_pass_rate")


def measure(song, candidate):
    texts = candidate["texts"]
    reviews = [review(song, s.id, Draft(s.id, tuple(
        DraftPhrase(p.id, texts.get(p.id, "")) for p in song.translatable_of(s.id)
    )), draft_run="evaluation") for s in song.sections]
    failing = set().union(*(r.failing_ids() for r in reviews))
    rows = []
    for p in song.phrases:
        covered = p.id in texts and bool(texts[p.id].strip())
        text = texts.get(p.id)
        n = count_syllables(text) if covered and p.translatable else None
        distance = max(p.budget.min - n, n - p.budget.max, 0) if p.budget is not None and n is not None else None
        rows.append({"id": p.id, "section": p.section, "keep_source": p.keep_source,
                     "text": text, "covered": covered, "actual": n,
                     "budget": p.budget.to_json() if p.budget else None,
                     "budget_distance": distance, "pattern": list(p.pattern) if p.pattern else None,
                     "explicit_pattern_match": ([count_syllables(x) for x in text.split() if count_syllables(x)] == list(p.pattern)) if p.pattern and n is not None else None,
                     "verdict": "skip" if p.keep_source else "missing" if not covered else "fail" if p.id in failing else "pass"})
    translated = [r for r in rows if not r["keep_source"]]
    covered = [r for r in translated if r["covered"]]
    budgeted = [r for r in covered if r["budget_distance"] is not None]
    return {
        "completed": candidate["completed"], "source_status": candidate["status"],
        "expected_phrases": len(translated), "covered_phrases": len(covered),
        "phrase_coverage": ratio(len(covered), len(translated)),
        "kept_phrases": sum(p.keep_source for p in song.phrases),
        "evaluated_phrases": len(budgeted),
        "budget_pass_rate": average([r["budget_distance"] == 0 for r in budgeted]),
        "mean_budget_distance": average([r["budget_distance"] for r in budgeted]),
        "under_budget_rate": average([r["actual"] < r["budget"]["min"] for r in budgeted]),
        "over_budget_rate": average([r["actual"] > r["budget"]["max"] for r in budgeted]),
        "rule_pass_rate": ratio(sum(r["verdict"] == "pass" for r in covered), len(covered)),
        "violations": dict(Counter(v.code for r in reviews for v in r.violations)),
        "issues": candidate["issues"], "error": candidate.get("error"), "phrases": rows,
    }


def _aggregate(rows, songs, variants=VARIANTS):
    result = {}
    for variant in variants:
        selected = [r for r in rows if r["song"] in songs and r["variant"] == variant]
        result[variant] = {
            "songs": len(selected), "completed_songs": sum(r["completed"] for r in selected),
            "song_completion_rate": average([r["completed"] for r in selected]),
            "expected_phrases": sum(r["expected_phrases"] for r in selected),
            "covered_phrases": sum(r["covered_phrases"] for r in selected),
            "phrase_coverage": ratio(sum(r["covered_phrases"] for r in selected), sum(r["expected_phrases"] for r in selected)),
            "evaluated_phrases": sum(r["evaluated_phrases"] for r in selected),
            **{k: average([r[k] for r in selected]) for k in METRICS},
            "valid_songs": {k: sum(r[k] is not None for r in selected) for k in METRICS},
        }
    return result


def bootstrap_difference(pairs, *, seed=0, samples=2000):
    """pairs 是逐歌 (a,b)，区间仅反映歌曲抽样，不模拟生成波动。"""
    if not pairs:
        return {"songs": 0, "difference_a_minus_b": None, "ci95": None}
    differences = [a - b for a, b in pairs]
    rng = random.Random(seed)
    draws = sorted(average([rng.choice(differences) for _ in differences]) for _ in range(samples))
    return {"songs": len(pairs), "difference_a_minus_b": average(differences),
            "ci95": [draws[int((samples - 1) * .025)], draws[int((samples - 1) * .975)]],
            "seed": seed, "samples": samples}


def evaluate_rules(snapshot, *, seed=0, samples=2000):
    variants = variants_of(snapshot)
    rows = []
    for item in snapshot["songs"]:
        song = Song.from_json(item["song"])
        for variant in variants:
            rows.append({"song": item["directory"], "variant": variant, **measure(song, item["candidates"][variant])})
    songs = [s["directory"] for s in snapshot["songs"]]
    paired = [s for s in songs if all(r["completed"] for r in rows if r["song"] == s)]
    differences = []
    for a, b in combinations(variants, 2):
        for metric in METRICS:
            pairs = []
            for sid in paired:
                ra = next(r for r in rows if r["song"] == sid and r["variant"] == a)
                rb = next(r for r in rows if r["song"] == sid and r["variant"] == b)
                if ra[metric] is not None and rb[metric] is not None:
                    pairs.append((ra[metric], rb[metric]))
            differences.append({"a": a, "b": b, "metric": metric, **bootstrap_difference(pairs, seed=seed, samples=samples)})
    return {"per_song": rows, "all_songs": _aggregate(rows, songs, variants),
            "paired_songs": paired, "paired": _aggregate(rows, paired, variants), "paired_differences": differences,
            "note": "宏平均为歌曲等权；缺失不填零，另报覆盖率；硬规则通过率不重复计入总分。"}
