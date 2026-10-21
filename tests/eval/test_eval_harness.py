"""The eval harness is itself tested, and its keyword-mode run is a regression gate."""
import json

from arbot.data.faq_store import FaqStore
from arbot.data.hash_embedder import HashEmbedder
from arbot.data.loaders import load_faq
from arbot.graph.bot import ChatBot
from eval.run import check_expect, main, percentile, retrieval_eval, run_scenario, substitute


def test_substitute_recurses_through_structures():
    out = substitute({"say": "${a}", "expect": {"contains": ["x ${b} y"]}}, {"a": "1", "b": "2"})
    assert out == {"say": "1", "expect": {"contains": ["x 2 y"]}}


def test_check_expect_reports_every_kind_of_failure():
    state = {"step": "A1_BUDGET", "phone": None}
    assert check_expect("hello there", state, {"equals": "hello there", "contains": ["hello"],
                                                "ends_with": "there", "state": {"phone": None}}) == []
    problems = check_expect("hello", state, {"equals": "bye", "contains": ["zzz"], "not_contains": ["hell"],
                                              "ends_with": "x", "state": {"step": "E1_PHONE"}})
    assert len(problems) == 5


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    assert percentile([10, 20, 30, 40, 50], 100) == 50


def test_a_failing_scenario_is_reported_with_the_offending_turn():
    scenario = {"id": "x", "turns": [{"say": "hello", "expect": {"equals": "Can I get your name?"}},
                                     {"say": "Sam", "expect": {"equals": "WRONG"}},
                                     {"say": "never reached"}]}
    result = run_scenario(ChatBot(), scenario, {})
    assert not result["passed"] and result["turns"] == 2
    assert result["failures"][0]["turn"] == 2 and result["failures"][0]["reply"].startswith("Thanks, Sam.")


def test_retrieval_eval_scores_hits_and_rejections():
    store = FaqStore(load_faq(), HashEmbedder())
    items = [{"q": "What is your head office address?", "expected": "FAQ-009"},
             {"q": "What is your head office address?", "expected": "FAQ-001"},
             {"q": "quantum chromodynamics lagrangian", "expected": None}]
    r = retrieval_eval(store, items, floor=0.3)
    assert r["answerable"] == 2 and r["recall_at_1"] == 0.5 and r["unanswerable_rejected"] == 1
    assert r["misses"] and r["misses"][0]["expected"] == "FAQ-001"


def test_keyword_mode_run_passes_every_scenario(tmp_path):
    report = tmp_path / "report.json"
    assert main(["--mode", "keyword", "--report", str(report)]) == 0
    data = json.loads(report.read_text())
    assert data["summary"]["pass_rate"] == 1.0 and data["summary"]["scenarios"] >= 25
    assert data["retrieval"]["recall_at_3"] >= 0.9
