"""实际演唱评分只能显式导入真人记录，纯文本评测不产生此分数。"""

from collections import defaultdict

from .common import average, read, variants_of

DIMENSIONS = ("melody_fit", "singing_difficulty")


def template():
    return {"protocol": "填写实际演唱/听评条件、评分锚点；旋律贴合1差5好，跟唱难度1容易5困难", "ratings": []}


def load_ratings(path, snapshot):
    variants = variants_of(snapshot)
    value = read(path)
    if not isinstance(value, dict) or not isinstance(value.get("protocol"), str) or not value["protocol"].strip() or not isinstance(value.get("ratings"), list):
        raise ValueError("真人记录须包含非空 protocol 和 ratings 数组")
    available = set()
    for s in snapshot["songs"]:
        for sec in s["song"]["sections"]:
            needed = {p["id"] for p in s["song"]["phrases"] if p["section"] == sec["id"] and not p.get("keep_source")}
            for variant in variants:
                if needed and needed <= s["candidates"][variant]["texts"].keys():
                    available.add((s["directory"], sec["id"], variant))
    seen = set()
    for row in value["ratings"]:
        if not isinstance(row, dict) or not isinstance(row.get("rater"), str) or not row["rater"].strip():
            raise ValueError("真人记录缺少 rater")
        if any(not isinstance(row.get(k), str) for k in ("song", "section", "variant")):
            raise ValueError("真人记录的 song、section、variant 必须为字符串")
        key = (row["song"], row["section"], row["variant"])
        if key not in available:
            raise ValueError("真人记录指向未知或缺失译词的段落")
        identity = (*key, row["rater"])
        if identity in seen:
            raise ValueError("同一评审对同一候选的真人记录重复")
        seen.add(identity)
        for dimension in DIMENSIONS:
            if type(row.get(dimension)) is not int or not 1 <= row[dimension] <= 5:
                raise ValueError("真人评分须为 1–5 的整数")
    summary = {}
    for variant in variants:
        summary[variant] = {}
        for dimension in DIMENSIONS:
            sections = defaultdict(list)
            for row in value["ratings"]:
                if row["variant"] == variant:
                    sections[(row["song"], row["section"])].append(row[dimension])
            songs = defaultdict(list)
            for (song, _), ratings in sections.items():
                songs[song].append(average(ratings))
            summary[variant][dimension] = {"macro_score": average([average(v) for v in songs.values()]),
                                           "valid_songs": len(songs), "rated_sections": len(sections)}
    return {"status": "completed" if value["ratings"] else "not_evaluated", **value, "summary": summary,
            "note": "导入的真人记录，尚未验证评审身份；稀疏覆盖时不可直接当作配对结论。"}
