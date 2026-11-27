"""Run bounded synthetic A/B arms through real translation backends.

API calls require an explicit dollar cap and prices. No user media or subtitle
files are read. The pre-change source snapshot supplies the baseline prompt and
repair contract; candidate arms use the current implementation. Reports retain
pre/post-repair texts, rather than confusing a clean JSON response with quality.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback

if str(_REPO_ROOT := Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools._common import PROJECT_ROOT as ROOT, ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from core.config import load_config

load_config()

from llm import engine, repair
from llm.backends import backend_lease
from llm.context import SourceContext
from llm.profiles.json_v3 import JsonProfile
from llm.run_context import RunContext
from llm.session import TranslationSession
from llm.output_checks import invalid_line_output
from tools.audits.translation_quality_cases import build_cases, plan_source_cases


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("missing baseline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def baseline_modules(root: Path):
    base = root / "src/llm"
    prompt = _load("quality_baseline_prompt", base / "prompt.py")
    json_profile = _load("quality_baseline_json", base / "profiles/json_v3.py")
    json_profile.prompt_module = prompt
    json_profile.JsonProfile.version = prompt.PROMPT_VERSION
    repair_module = _load("quality_baseline_repair", base / "repair.py")
    repair_module.prompt_module = prompt
    repair_module.json_v3 = json_profile
    baseline_engine = _load("quality_baseline_engine", base / "engine.py")
    return json_profile.JsonProfile(), repair_module, baseline_engine


class SpendLimit(RuntimeError):
    pass


class Budget:
    def __init__(self, *, maximum: float, input_price: float, output_price: float, requests: int):
        self.maximum, self.input_price, self.output_price = maximum, input_price / 1e6, output_price / 1e6
        self.limit, self.reserved, self.requests = requests, 0.0, 0
        self.lock = threading.Lock()
        self.usage: list[dict] = []

    def reserve(self, messages, max_tokens, *, paid):
        # UTF-8 bytes pessimistically bound text token count; extra allowance
        # covers role wrappers and schema framing. No refund for interrupted
        # responses whose actual consumption is unknown.
        prompt_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 8192
        cost = (prompt_bound * self.input_price + max_tokens * self.output_price) if paid else 0.0
        with self.lock:
            if self.requests >= self.limit or self.reserved + cost > self.maximum:
                raise SpendLimit("experiment request/cost bound reached")
            self.reserved += cost
            self.requests += 1
            return self.requests


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise ValueError("output already contains an experiment report; use a fresh directory")
    if args.backend == "api" and (args.max_usd is None or args.max_usd <= 0):
        raise ValueError("API experiments require --max-usd")
    if args.backend == "api" and args.priced_model != os.getenv("LLM_MODEL_NAME", ""):
        raise ValueError("--priced-model must match the configured model")
    if args.backend == "llamacpp":
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        if args.server:
            os.environ["LLAMACPP_SERVER_PATH"] = str(Path(args.server).resolve())
        from llm.backends.llamacpp_server import resolve_gguf_model_path

        os.environ["LLAMACPP_GGUF_PATH"] = resolve_gguf_model_path(download_enabled=False)
    backend_name = "openai" if args.backend == "api" else "llamacpp"
    os.environ["TRANSLATION_BACKEND"] = backend_name
    # The experiment accounts for each attempt itself. Disable SDK retries so
    # an interrupted paid request cannot invisibly multiply its reserved cost.
    if args.backend == "api":
        from llm.backends import openai_compat

        factory = openai_compat._make_async_client

        def no_sdk_retries():
            client = factory()
            client.max_retries = 0
            return client

        openai_compat._make_async_client = no_sdk_retries
    base_json, base_repair, base_engine = baseline_modules(Path(args.baseline))
    all_cases = build_cases()
    # Evenly distributed scenes make a small smoke cover both ends of the set.
    scene_count = min(len(all_cases) // 4, max(1, args.cues // 4))
    scene_indexes = [index * (len(all_cases) // 4) // scene_count for index in range(scene_count)]
    cases = [row for scene in scene_indexes for row in all_cases[scene * 4:scene * 4 + 4]]
    original_cue_count = len(cases)
    cases = plan_source_cases(cases)
    input_kind = "source_sentence_plan"
    segments = [{key: value for key, value in row.items() if key not in {"reference", "category", "case_id", "checkpoint"}} for row in cases]
    context = SourceContext.build(segments)
    full_payload = json.dumps(list(context.rows), ensure_ascii=False, separators=(",", ":"))
    budget = Budget(maximum=args.max_usd or 0.0, input_price=args.input_price or 0.0, output_price=args.output_price or 0.0, requests=args.max_requests)
    report = {
        "schema": "translation_quality_experiment_v2", "synthetic_only": True,
        "input_kind": input_kind, "source_timing": "synthetic fixture; no audio alignment measured",
        "original_cue_count": original_cue_count,
        "source_signature": context.signature, "backend": args.backend,
        # File name only: an explicit GGUF path would put a local directory in the report.
        "configured_model": os.getenv("LLM_MODEL_NAME", "") if args.backend == "api" else Path(
            os.getenv("LLAMACPP_GGUF_PATH", "") or os.getenv("LLAMACPP_MODEL_FILE", "")
        ).name,
        "glossary": args.glossary, "cue_count": len(cases), "cases": cases, "arms": {},
        "maximum_usd": args.max_usd, "prices_per_million": {"input": args.input_price, "output": args.output_price},
    }
    request_dir = out / "requests"
    request_dir.mkdir()

    def save():
        report["requests"] = budget.requests
        report["reserved_upper_bound_usd"] = round(budget.reserved, 8)
        report["usage"] = budget.usage
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with backend_lease(backend_name) as backend:
        for arm in args.arms:
            if arm == "baseline":
                profile = base_json
            elif arm == "sampling":
                profile = type("OldSamplingJson", (JsonProfile,), {
                    "version": "json-old-sampling", "sampling_parameters": lambda self: {},
                })()
            else:
                profile = JsonProfile()
            is_local = args.backend == "llamacpp"
            session = TranslationSession(
                backend_name=backend_name, profile=profile,
                batch_size=min(args.batch_size, 16) if is_local else args.batch_size,
                max_workers=args.workers, cache_path="", target_lang="简体中文",
                glossary=args.glossary, character_reference="",
                reasoning_effort="" if is_local else "low",
            )
            arm_started = time.perf_counter()
            arm_report = {"profile": profile.cache_signature(), "sampling": profile.sampling_parameters(), "status": "running"}
            report["arms"][arm] = arm_report
            save()
            with session.bound():
                run_context = RunContext.capture()

                def chat(messages, *, expected_count=0, max_tokens=None, response_schema=None, bounded_response_schema=None,
                         reasoning_effort=None, on_usage=None, on_progress=None, cancel_event=None, output_grammar=None,
                         run_context=run_context, profile=profile, arm=arm):
                    run_context.adopt()
                    bound = min(args.max_output_tokens, max_tokens or args.max_output_tokens)
                    request_id = budget.reserve(messages, bound, paid=args.backend == "api")
                    sampling = profile.sampling_parameters()
                    extra = {"sampling_parameters": sampling} if args.backend == "llamacpp" and sampling else {}
                    if args.backend == "llamacpp" and output_grammar is not None:
                        extra["grammar"] = output_grammar

                    def usage(row):
                        with budget.lock:
                            budget.usage.append({"arm": arm, **row})
                        if on_usage:
                            on_usage(row)

                    trace = {"arm": arm, "request_id": request_id, "messages": messages,
                             "schema": bounded_response_schema or response_schema, "max_tokens": bound,
                             "grammar": extra.get("grammar")}
                    try:
                        response = backend.chat_completion(
                            messages, expected_count=expected_count,
                            temperature=float(sampling.get("temperature", 0.6)), top_p=float(sampling.get("top_p", 0.9)),
                            max_tokens=bound, response_format=bounded_response_schema or response_schema,
                            reasoning_effort=reasoning_effort or "low", on_usage=usage,
                            on_progress=on_progress, cancel_event=cancel_event, **extra,
                        )
                        trace["response"] = response
                        return response
                    except Exception as exc:
                        trace["error_type"] = type(exc).__name__
                        raise
                    finally:
                        (request_dir / f"request-{request_id:04d}.json").write_text(
                            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8",
                        )

                try:
                    runner = base_engine if arm == "baseline" else engine
                    context_kwargs = {} if arm == "baseline" else {"source_context": context}
                    texts, timings, retries = runner.run_batched(
                        segments, profile=profile, backend_name=backend_name, chat=chat,
                        backoff_sleep=lambda *_a, **_k: None, crash_probe=lambda: 0,
                        batch_size=session.batch_size, max_workers=args.workers,
                        api_retries=2, batch_repair_retries=1, batch_max_requests=4,
                        prefix_warmup=False, extra_glossary="", full_context="",
                        full_source_payload="" if is_local else full_payload,
                        use_full_json_prefix=not is_local,
                        cache_path="", cache_lock=threading.Lock(), target_lang=session.target_lang,
                        glossary=session.glossary, character_reference="", prompt_version=profile.cache_signature(),
                        model_identity=backend.cache_identity(), compact_system_prompt=is_local,
                        reasoning_effort=session.reasoning_effort, **context_kwargs,
                    )
                    arm_report["first_pass"] = list(texts)
                    arm_report.update({"timings": timings, "retries": retries})
                    save()
                    if profile.wants_repair_pass:
                        module = base_repair if arm == "baseline" else repair
                        texts, timing = module.apply_repair_pass(segments, texts, chat=chat, session=session, extra_glossary="")
                        if timing:
                            timings.append(timing)
                    arm_report["after_repair"] = list(texts)
                    arm_report.update({
                        "texts": texts, "status": "complete",
                        "remaining_kana_or_empty_ids": [index for index, (seg, target) in enumerate(zip(segments, texts))
                            if invalid_line_output(seg["text"], target, session.target_lang)],
                    })
                except Exception as exc:
                    # Keep diagnostic locations, never arbitrary exception text
                    # that a transport may populate with endpoint credentials.
                    frames = []
                    for frame in traceback.extract_tb(exc.__traceback__):
                        path = Path(frame.filename)
                        name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
                        frames.append({"file": name, "line": frame.lineno, "function": frame.name})
                    arm_report.update({"status": "failed", "error_type": type(exc).__name__, "error_frames": frames})
                    if isinstance(exc, SpendLimit):
                        save()
                        break
                finally:
                    arm_report["elapsed_s"] = round(time.perf_counter() - arm_started, 3)
                    save()
                    print(json.dumps({"arm": arm, "status": arm_report["status"], "elapsed_s": arm_report["elapsed_s"],
                                      "requests": budget.requests, "reserved_upper_bound_usd": round(budget.reserved, 6)}), flush=True)
    return 0 if all(arm["status"] == "complete" for arm in report["arms"].values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("api", "llamacpp"), required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="+", choices=("baseline", "candidate", "sampling"), default=["baseline", "candidate"])
    parser.add_argument("--cues", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-requests", type=int, default=50)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--output-price", type=float)
    parser.add_argument("--priced-model", default="")
    parser.add_argument("--glossary", default="")
    parser.add_argument("--server", default="")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.cues, args.workers, args.batch_size, args.max_output_tokens, args.max_requests)):
        parser.error("counts must be positive")
    if args.backend == "api" and any(value is None or not math.isfinite(value) or value <= 0 for value in (args.max_usd, args.input_price, args.output_price)):
        parser.error("API experiments require finite positive --max-usd, --input-price and --output-price")
    if args.backend == "api" and any(arm not in {"baseline", "candidate"} for arm in args.arms):
        parser.error("native sampling/context ablations require --backend llamacpp")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
