"""Scripted-conversation evaluation: pass rate, latency, tokens, cost and FAQ retrieval recall.

    python -m eval.run --mode keyword          # no API key, deterministic, runs in CI
    python -m eval.run --mode auto --delay 2   # real LLM from .env (throttled for free tiers)

Exit code is 1 if the scenario pass rate is below --min-pass-rate.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import yaml

from arbot.app import build_bot, build_embedder_with_fallback
from arbot.config import Settings
from arbot.data.faq_store import FaqStore
from arbot.data.hash_embedder import HashEmbedder
from arbot.data.loaders import load_faq
from eval.data_vars import variables
from arbot.services.observability import ListObserver

HERE = Path(__file__).parent
CONSTANTS: dict[str, str] = {}


def substitute(value, vars_: dict[str, str]):
    """Replace ${name} in strings, recursively through lists and dicts."""
    if isinstance(value, str):
        return re.sub(r"\$\{(\w+)\}", lambda m: vars_[m.group(1)], value)
    if isinstance(value, list):
        return [substitute(v, vars_) for v in value]
    if isinstance(value, dict):
        return {k: substitute(v, vars_) for k, v in value.items()}
    return value


def check_expect(reply: str, state: dict, expect: dict) -> list[str]:
    """Return a list of human-readable failures (empty = pass)."""
    failures = []
    if "equals" in expect and reply != expect["equals"]:
        failures.append(f"expected reply == {expect['equals']!r}")
    for text in expect.get("contains", []):
        if text not in reply:
            failures.append(f"reply should contain {text!r}")
    if "contains_any" in expect and not any(t in reply for t in expect["contains_any"]):
        failures.append(f"reply should contain one of {expect['contains_any']!r}")
    for text in expect.get("not_contains", []):
        if text in reply:
            failures.append(f"reply should not contain {text!r}")
    if "ends_with" in expect and not reply.endswith(expect["ends_with"]):
        failures.append(f"reply should end with {expect['ends_with']!r}")
    for key, want in expect.get("state", {}).items():
        if state.get(key) != want:
            failures.append(f"state[{key}] expected {want!r}, got {state.get(key)!r}")
    return failures


def run_scenario(bot, scenario: dict, vars_: dict, delay: float = 0.0) -> dict:
    tid = bot.new_thread_id()
    failures, turns_run = [], 0
    for i, turn in enumerate(substitute(scenario["turns"], vars_), 1):
        if delay and i > 1:
            time.sleep(delay)
        result = bot.send(tid, turn["say"])
        turns_run += 1
        problems = check_expect(result.reply, result.state, turn.get("expect", {}))
        if problems:
            failures.append({"turn": i, "said": turn["say"], "reply": result.reply, "problems": problems})
            break                                    # later turns depend on this one
    return {"id": scenario["id"], "passed": not failures, "turns": turns_run, "failures": failures,
            "tags": scenario.get("tags", [])}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def retrieval_eval(store: FaqStore, items: list[dict], floor: float) -> dict:
    hits1 = hits3 = answerable = rr = 0.0
    rejected_ok = unanswerable = 0
    misses = []
    for item in items:
        results = store.search(item["q"], k=3)
        if item["expected"] is None:
            unanswerable += 1
            if not results or results[0].score < floor:
                rejected_ok += 1
            else:
                misses.append({"q": item["q"], "expected": None, "got": results[0].faq_id,
                               "score": round(results[0].score, 3)})
            continue
        answerable += 1
        ids = [r.faq_id for r in results]
        hits1 += ids[:1] == [item["expected"]]
        hits3 += item["expected"] in ids
        rr += 1 / (ids.index(item["expected"]) + 1) if item["expected"] in ids else 0
        if item["expected"] not in ids:
            misses.append({"q": item["q"], "expected": item["expected"], "got": ids})
    return {
        "answerable": int(answerable), "recall_at_1": round(hits1 / answerable, 3),
        "recall_at_3": round(hits3 / answerable, 3), "mrr": round(rr / answerable, 3),
        "unanswerable": unanswerable, "unanswerable_rejected": rejected_ok,
        "misses": misses,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["auto", "keyword"], default="auto",
                    help="auto = LLM from env (falls back to rules without a key); keyword = never use an LLM")
    ap.add_argument("--only", help="run only scenarios whose id contains this text")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between turns (free-tier rate limits)")
    ap.add_argument("--min-pass-rate", type=float, default=1.0)
    ap.add_argument("--report", default=str(HERE / "reports" / "latest.json"))
    args = ap.parse_args(argv)

    settings = Settings(lead_db_path="", turn_log_path="", chroma_dir="", checkpoint_db="")
    if args.mode == "keyword":
        settings.llm_api_key = ""
        embedder = HashEmbedder()
    else:
        embedder = build_embedder_with_fallback(settings)
    observer = ListObserver()
    bot = build_bot(settings, embedder=embedder, observer=observer, on_lead=lambda s: None, callbacks=[])
    llm_active = hasattr(bot.ctx.nlu, "chain")

    vars_ = variables()
    scenarios = yaml.safe_load((HERE / "scenarios" / "scenarios.yaml").read_text(encoding="utf-8"))
    skipped = [s["id"] for s in scenarios if s.get("llm_only") and not llm_active]
    scenarios = [s for s in scenarios if s["id"] not in skipped and (not args.only or args.only in s["id"])]

    results = [run_scenario(bot, s, vars_, args.delay) for s in scenarios]
    passed = sum(r["passed"] for r in results)
    rate = passed / len(results) if results else 0.0

    latencies = [r["latency_ms"] for r in observer.records]
    stats = {
        "mode": "llm" if llm_active else "rule-based",
        "embedder": getattr(embedder, "model_name", type(embedder).__name__),
        "scenarios": len(results), "passed": passed, "pass_rate": round(rate, 3), "skipped": skipped,
        "turns": len(observer.records),
        "latency_ms_p50": round(percentile(latencies, 50), 1), "latency_ms_p95": round(percentile(latencies, 95), 1),
        "latency_ms_mean": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "llm_calls": sum(r["llm_calls"] for r in observer.records),
        "input_tokens": sum(r["input_tokens"] for r in observer.records),
        "output_tokens": sum(r["output_tokens"] for r in observer.records),
        "est_cost_usd": round(sum(r["est_cost_usd"] for r in observer.records), 6),
        "degraded_turns": sum(r["degraded"] for r in observer.records),
    }
    retrieval_items = yaml.safe_load((HERE / "retrieval_set.yaml").read_text(encoding="utf-8"))
    floor = getattr(embedder, "thresholds", (0, settings.faq_min_threshold))[1]
    retrieval = retrieval_eval(FaqStore(load_faq(), embedder), retrieval_items, floor)

    report = {"summary": stats, "retrieval": retrieval, "results": results}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nScenarios: {passed}/{len(results)} passed ({rate:.0%})   mode={stats['mode']}   "
          f"embedder={stats['embedder']}")
    for r in results:
        if not r["passed"]:
            f = r["failures"][0]
            print(f"  FAIL {r['id']} (turn {f['turn']}: {f['said']!r}) -> {'; '.join(f['problems'])}")
            print(f"       reply: {f['reply']!r}")
    if skipped:
        print(f"  skipped (need an LLM): {', '.join(skipped)}")
    print(f"Latency per turn: p50={stats['latency_ms_p50']}ms p95={stats['latency_ms_p95']}ms | "
          f"LLM calls={stats['llm_calls']} tokens in/out={stats['input_tokens']}/{stats['output_tokens']} "
          f"est. cost=${stats['est_cost_usd']} | degraded turns={stats['degraded_turns']}")
    print(f"FAQ retrieval: recall@1={retrieval['recall_at_1']} recall@3={retrieval['recall_at_3']} "
          f"MRR={retrieval['mrr']} | unanswerable rejected {retrieval['unanswerable_rejected']}/"
          f"{retrieval['unanswerable']}")
    for m in retrieval["misses"]:
        print(f"  retrieval miss: {m}")
    print(f"Report written to {args.report}")
    return 0 if rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
