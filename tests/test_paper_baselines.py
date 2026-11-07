"""论文 Exp.4 两组 baseline：原样组与人工标注组，生成 + 同模型清理。"""

import json

import pytest

from baseline.run_baselines import (
    BaselineOutputError,
    PaperClient,
    align_lines,
    build_paper_request,
    main,
    parse_lines,
    run_paper,
)
from dubbing.llm import LLMError, ScriptedClient
from dubbing.models import Song
from dubbing.store import RunStore

from helpers import phrase, song_json, write_song


def _song():
    return Song.from_json(
        song_json(
            "test",
            ["V1", "C1"],
            [
                {"id": "V1-01", "section": "V1", "ja": "あ", "budget": {"min": 12, "max": 13}, "pattern": [6, 6]},
                phrase("V1-02", "V1", "La la", keep_source=True),
                phrase("C1-01", "C1", "い", 4),
            ],
        )
    )


def _script():
    return [
        "第一段原始输出，包含解释。",
        json.dumps({"lines": ["一二三四五六七八九十天地", "随便"]}, ensure_ascii=False),
        "第二段原始输出。",
        json.dumps({"lines": ["风雨晴空"]}, ensure_ascii=False),
    ]


def test_original_input_is_invariant_to_human_annotations():
    song = _song()
    changed = song.to_json()
    changed["phrases"][0].update(budget={"min": 8, "max": 8}, pattern=[3, 5], notes="人工备注")
    changed["phrases"][1].update(keep_source=False, budget={"min": 2, "max": 3})
    before = build_paper_request(song, "V1", "paper-original")
    after = build_paper_request(Song.from_json(changed), "V1", "paper-original")
    assert before == after
    assert before.user.splitlines() == ["あ", "La la"]
    assert build_paper_request(song, "C1", "paper-original").user == "い"


def test_annotations_preserve_interval_pattern_and_kept_source():
    request = build_paper_request(_song(), "V1", "paper-annotated")
    rows = json.loads(request.user)["phrases"]
    assert rows[0]["budget"] == {"min": 12, "max": 13}
    assert rows[0]["pattern"] == [6, 6]
    assert rows[1]["keep_source"] is True
    assert rows[1]["budget"] is None
    assert [p["id"] for p in rows] == ["V1-01", "V1-02"]


@pytest.mark.parametrize("variant", ["paper-original", "paper-annotated"])
def test_runs_by_section_cleans_aligns_and_writes_lyrics(tmp_path, variant):
    client = ScriptedClient(_script())
    out = tmp_path / variant
    run_dir = run_paper(_song(), variant, client, out=out)

    # 每段生成一次、清理一次；没有规则返修
    assert len(client.calls) == 4
    assert client.calls[1]["user"] == "第一段原始输出，包含解释。"
    assert client.calls[3]["user"] == "第二段原始输出。"

    store = RunStore(out)
    run_id = store.latest_run()
    assert run_dir == store.run_dir(run_id)
    generation = store.read_output(run_id, "generation", 1)
    assert generation["section"] == "V1"
    assert generation["source"] == "cleanup.lines"
    assert [p["id"] for p in generation["phrases"]] == ["V1-01", "V1-02"]
    assert store.read_output(run_id, "cleanup", 1)["lines"] == ["一二三四五六七八九十天地", "随便"]
    for agent in ("generation", "cleanup"):
        names = sorted(p.name for p in store.record_dir(run_id, agent, 1).iterdir())
        assert names == ["meta.json", "output.json", "request.json"]
    assert (run_dir / "lyrics.zh.txt").is_file()
    assert (run_dir / "lyrics.md").is_file()
    assert store.read_latest()["status"] == "completed"


def test_rule_failure_is_reported_without_repair_or_changing_lyrics(tmp_path):
    responses = _script()
    responses[1] = json.dumps({"lines": ["少字", "随便"]})
    client = ScriptedClient(responses)
    out = tmp_path / "paper-annotated"
    run_dir = run_paper(_song(), "paper-annotated", client, out=out)

    assert len(client.calls) == 4  # 预算不过也不返修
    text = (run_dir / "lyrics.zh.txt").read_text(encoding="utf-8")
    assert "少字" in text
    md = (run_dir / "lyrics.md").read_text(encoding="utf-8")
    assert "fail" in md


def test_extra_line_is_failure_instead_of_silently_dropping_title(tmp_path):
    client = ScriptedClient(["原始歌词", json.dumps({"lines": ["疑似标题", "第一句", "第二句"]})])
    out = tmp_path / "paper-original"
    with pytest.raises(BaselineOutputError, match="3 行，预期 2 行"):
        run_paper(_song(), "paper-original", client, out=out)

    store = RunStore(out)
    run_id = store.latest_run()
    assert store.read_meta(run_id, "generation", 1)["raw_response"] == "原始歌词"
    assert "abandoned" in store.read_meta(run_id, "cleanup", 1)
    assert not (store.run_dir(run_id) / "lyrics.zh.txt").exists()
    assert len(client.calls) == 2


@pytest.mark.parametrize("payload", [{"lines": [""]}, {"lines": [3]}, {"other": []}, ["译文"]])
def test_cleanup_rejects_unusable_line_mapping(payload):
    with pytest.raises(BaselineOutputError):
        parse_lines(json.dumps(payload), 1)


def test_dry_runs_are_isolated_and_do_not_touch_existing_products(root, monkeypatch):
    write_song(root, "demo", ["S1"], [phrase("P1", "S1", "あ", 4)])
    directory = root / "demo"
    song_before = (directory / "song.yaml").read_text(encoding="utf-8")
    # 旧布局的遗留文件：新代码不该碰它们
    (directory / "baseline.txt").write_text("legacy", encoding="utf-8")

    def forbidden_call(*args, **kwargs):
        raise AssertionError("dry run must not contact the API")
    monkeypatch.setattr(PaperClient, "complete", forbidden_call)
    monkeypatch.setenv("DUBBING_MODEL", "current-project-model")
    monkeypatch.setenv("DUBBING_TEMPERATURE", "0.4")
    for _ in range(2):
        assert main(["--root", str(root), "--all", "--dry-run"]) == 0
    assert len(list(directory.glob("baselines/*/*/sections/001/generation/request.json"))) == 4
    assert len(list(directory.glob("baselines/one-shot/*/sections/001/B/request.json"))) == 2
    assert (directory / "song.yaml").read_text(encoding="utf-8") == song_before
    assert (directory / "baseline.txt").read_text(encoding="utf-8") == "legacy"
    assert not list(directory.glob("baselines/**/response.txt"))
    assert not list(directory.glob("baselines/**/output.json"))


def test_paid_calls_require_explicit_api_flag(root):
    write_song(root, "demo", ["S1"], [phrase("P1", "S1", "あ", 4)])
    assert main(["--root", str(root), "--all"]) == 2
    assert not list(root.rglob("baselines"))


def test_both_variants_share_current_model_and_save_call_usage(root, monkeypatch):
    write_song(root, "demo", ["S1"], [phrase("P1", "S1", "あ", 4)])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-secret-do-not-persist")
    monkeypatch.setenv("DUBBING_MODEL", "configured-current-model")
    monkeypatch.setenv("DUBBING_TEMPERATURE", "0.4")
    responses = iter(["原样生成", '{"lines":["风雨晴空"]}', "标注生成", '{"lines":["看风雨"]}'])
    seen = []
    class Response:
        status_code = 200
        def __init__(self, text):
            self.payload = {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}}
        def json(self):
            return self.payload
    def post(url, json, headers, timeout):
        seen.append(json)
        return Response(next(responses))
    monkeypatch.setattr("dubbing.llm.requests.post", post)
    assert main(["--root", str(root), "--all", "--baseline", "paper", "--client", "api"]) == 0
    assert len(seen) == 4
    assert {body["model"] for body in seen} == {"configured-current-model"}
    assert {body["temperature"] for body in seen} == {0.4}
    assert seen[0]["messages"][0]["content"][0]["text"] == "あ"
    annotated = json.loads(seen[2]["messages"][0]["content"][0]["text"])
    assert annotated["phrases"][0]["budget"] == {"min": 4, "max": 4}

    for meta_file in (root / "demo/baselines").glob("*/*/sections/001/*/meta.json"):
        meta = json.loads(meta_file.read_text())
        assert meta["model"] == "configured-current-model"
        assert meta["usage"]["output_tokens"] == 5


def test_api_failure_is_recorded_and_no_api_key_is_written(tmp_path, monkeypatch):
    payload = {"content": [{"type": "thinking", "thinking": "fixture"}], "stop_reason": "max_tokens", "usage": {"output_tokens": 10}}
    class Response:
        status_code = 200
        def json(self):
            return payload
    monkeypatch.setattr("dubbing.llm.requests.post", lambda *args, **kwargs: Response())
    client = PaperClient(api_key="private-test-key", model="current-model", max_retries=1)
    out = tmp_path / "paper-original"
    with pytest.raises(LLMError, match="max_tokens"):
        run_paper(_song(), "paper-original", client, out=out)

    store = RunStore(out)
    meta = store.read_meta(store.latest_run(), "generation", 1)
    assert "abandoned" in meta
    assert all("private-test-key" not in p.read_text() for p in out.rglob("*") if p.is_file())


def test_openai_gateway_protocol_and_usage(monkeypatch):
    seen = {}
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "译词", "reasoning_content": "fixture"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    def post(url, json, headers, timeout):
        seen.update(url=url, body=json, headers=headers)
        return Response()
    monkeypatch.setattr("dubbing.llm.requests.post", post)
    client = PaperClient(api_key="test-key", model="DeepSeek-V4.1-Flash", base_url="https://maas-api.ebcloud.com/v1", protocol="openai")
    assert client.complete(agent="baseline", system="instruction", user="lyrics") == "译词"
    assert seen["url"] == "https://maas-api.ebcloud.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer test-key"
    assert "x-api-key" not in seen["headers"]
    assert seen["body"]["messages"] == [{"role": "system", "content": "instruction"}, {"role": "user", "content": "lyrics"}]
    assert client.last_usage == {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.mark.parametrize("choice, error", [
    ({"message": {"content": "半截"}, "finish_reason": "length"}, "截断"),
    ({"message": {"content": None, "reasoning_content": "fixture"}, "finish_reason": "stop"}, "没有正文"),
])
def test_openai_rejects_unusable_responses(choice, error):
    class Response:
        def json(self):
            return {"choices": [choice]}
    client = PaperClient(protocol="openai")
    with pytest.raises(LLMError, match=error):
        client._text_of(Response(), agent="baseline")


def test_parallel_cli_keeps_results_in_directory_alias(root):
    import shutil
    write_song(root, "internal-id", ["S1"], [phrase("P1", "S1", "あ", 4)])
    shutil.move(str(root / "internal-id"), str(root / "directory-name"))
    assert main(["--root", str(root), "--all", "--baseline", "one-shot", "--jobs", "2", "--dry-run"]) == 0
    assert len(list((root / "directory-name").glob("baselines/one-shot/*/sections/001/B/request.json"))) == 1
    assert not (root / "internal-id").exists()


def test_structured_generation_is_not_lost_when_cleanup_returns_empty():
    generated = json.dumps({"phrases": [{"id": "P2", "zh": "月光"}, {"id": "P1", "zh": "夜色"}]})
    lines, source = align_lines(generated, '{"lines":[]}', ["P1", "P2"])
    assert lines == ["夜色", "月光"]
    assert source == "generation.phrases.zh"


def test_structured_generation_rejects_duplicate_ids():
    generated = json.dumps({"phrases": [{"id": "P1", "zh": "夜色"}, {"id": "P1", "zh": "月光"}]})
    with pytest.raises(BaselineOutputError, match="ID 重复"):
        align_lines(generated, '{"lines":["夜色","月光"]}', ["P1", "P2"])


def test_thinking_disabled_is_sent_and_ignored_switch_is_rejected(monkeypatch):
    bodies = []
    class Response:
        status_code = 200
        def json(self):
            return {"choices": [{"message": {"content": "译词", "reasoning_content": "fixture"}, "finish_reason": "stop"}], "usage": {"completion_tokens_details": {"reasoning_tokens": 1}}}
    def post(url, json, headers, timeout):
        bodies.append(json)
        return Response()
    monkeypatch.setattr("dubbing.llm.requests.post", post)
    client = PaperClient(api_key="test-key", protocol="openai", thinking="disabled")
    with pytest.raises(LLMError, match="网关仍返回推理"):
        client.complete(agent="baseline", system="prompt", user="lyrics")
    assert bodies[0]["enable_thinking"] is False
    assert client.last_thinking_observation["reasoning_tokens"] == 1


@pytest.mark.parametrize("reasoning", [False, True])
def test_anthropic_thinking_disabled_request_and_response_check(monkeypatch, reasoning):
    bodies = []
    class Response:
        status_code = 200
        def json(self):
            content = [{"type": "text", "text": "译词"}]
            if reasoning:
                content.insert(0, {"type": "thinking", "thinking": "fixture"})
            return {"content": content, "stop_reason": "end_turn", "usage": {"output_tokens": 5}}
    def post(url, json, headers, timeout):
        bodies.append(json)
        return Response()
    monkeypatch.setattr("dubbing.llm.requests.post", post)
    client = PaperClient(api_key="test-key", protocol="anthropic", thinking="disabled")
    if reasoning:
        with pytest.raises(LLMError, match="接口仍返回推理"):
            client.complete(agent="baseline", system="prompt", user="lyrics")
    else:
        assert client.complete(agent="baseline", system="prompt", user="lyrics") == "译词"
    assert bodies[0]["thinking"] == {"type": "disabled"}
    assert client.last_thinking_observation["reasoning_content_present"] is reasoning
