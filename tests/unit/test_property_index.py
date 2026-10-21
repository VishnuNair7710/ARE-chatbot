import pytest

from arbot.data.loaders import load_listings
from arbot.data.property_index import PropertyIndex, describe_property


@pytest.fixture(scope="module")
def idx():
    return PropertyIndex(load_listings())


def test_exact_name_and_address_find_one_home(idx):
    hits = idx.find("Apartment 3, Aldermoor Place")
    assert [h.property_id for h in hits] == ["AR-0003"]


def test_noise_words_and_typos_still_match(idx):
    assert idx.find("tell me about apartment 3 aldermoor place please")[0].property_id == "AR-0003"
    assert idx.find("Apartmnt 3 Aldermor Place")[0].property_id == "AR-0003"


def test_distinguishing_number_is_decisive(idx):
    assert idx.find("Apartment 2 Aldermoor Place")[0].property_id == "AR-0002"


def test_ambiguous_query_returns_several_for_the_visitor_to_choose(idx):
    hits = idx.find("The Old Rectory")
    assert {h.property_id for h in hits} == {"AR-0017", "AR-0080"}
    assert len(idx.find("Aldermoor Place")) > 1


def test_address_disambiguates_duplicate_names(idx):
    assert [h.property_id for h in idx.find("The Old Rectory, Sheen Road")] == ["AR-0080"]


def test_id_lookup_and_unknown(idx):
    assert idx.find("what about ar-0017?")[0].property_id == "AR-0017"
    assert idx.find("zzzz qqqq") == [] and idx.find("") == []


def test_get_returns_plain_python_values_and_none_for_unknown(idx):
    p = idx.get("ar-0001")
    assert p["bedrooms"] == 2 and isinstance(p["price_gbp"], int) and p["status"] == "available"
    assert idx.get("AR-9999") is None


def test_describe_property_is_factual(idx):
    text = describe_property(idx.get("AR-0001"))
    assert "£1,330,000" in text and "2 bedrooms" in text and "available" in text
