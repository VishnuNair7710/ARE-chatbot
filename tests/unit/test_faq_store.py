import pytest

from arbot.data.faq_store import FaqStore
from arbot.data.hash_embedder import HashEmbedder
from arbot.data.loaders import load_faq


@pytest.fixture(scope="module")
def store():
    return FaqStore(load_faq(), HashEmbedder())


@pytest.mark.parametrize("query,expected_id", [
    ("What is your head office address?", "FAQ-009"),
    ("where is the head office located", "FAQ-009"),
    ("What are the Chelsea branch opening hours?", "FAQ-023"),
    ("what time does the Oxford branch open", "FAQ-028"),
    ("do you charge buyers a fee", "FAQ-032"),
    ("how do I arrange a viewing", "FAQ-031"),
    ("is the valuation free", "FAQ-039"),
    ("do you do lettings", "FAQ-044"),
])
def test_retrieval_finds_the_right_entry_in_top3(store, query, expected_id):
    ids = [h.faq_id for h in store.search(query, k=3)]
    assert expected_id in ids, ids


def test_results_carry_stored_answer_and_scores_in_order(store):
    hits = store.search("What is your head office phone number?", k=3)
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert "020 7946 0100" in hits[0].answer


def test_unrelated_query_scores_low(store):
    assert store.search("quantum chromodynamics lagrangian", k=1)[0].score < 0.2


def test_persistent_index_is_reused_and_rebuilt_on_change(tmp_path):
    faq = load_faq()
    first = FaqStore(faq, HashEmbedder(), path=str(tmp_path))
    assert first._col.count() == len(faq)
    again = FaqStore(faq, HashEmbedder(), path=str(tmp_path))
    assert again._col.name == first._col.name
    changed = faq.copy()
    changed.loc[0, "answer"] = "A different answer"
    third = FaqStore(changed, HashEmbedder(), path=str(tmp_path))
    assert third._col.name != first._col.name
    assert third.search("Who are Ashcombe & Rowe Estates?", k=1)[0].answer == "A different answer"
