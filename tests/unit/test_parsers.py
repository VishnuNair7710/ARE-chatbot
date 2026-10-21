import pytest
from hypothesis import given, strategies as st

from arbot.domain.budget import parse_budget
from arbot.domain.postcode import extract_postcode, normalise_postcode
from arbot.domain.validators import validate_email_address, validate_phone


@pytest.mark.parametrize("text,expected", [
    ("£1.5m", 1_500_000),
    ("1,200,000", 1_200_000),
    ("1.2 million", 1_200_000),
    ("950k", 950_000),
    ("£950K", 950_000),
    ("my budget is around £2m please", 2_000_000),
    ("a million", 1_000_000),
    ("half a million", 500_000),
    ("1000000", 1_000_000),
    ("I want 3 bedrooms and £1.2m", 1_200_000),
    ("somewhere around one point five million", 1_500_000),
    ("two and a half million", 2_500_000),
    ("a million and a half", 1_500_000),
    ("nine hundred and fifty thousand", 950_000),
    ("one million two hundred thousand", 1_200_000),
    ("sixty-five thousand pounds", 65_000),
    ("I need three bedrooms, budget one point two million", 1_200_000),
])
def test_budget_parses(text, expected):
    r = parse_budget(text)
    assert r.value == expected and not r.ambiguous


@pytest.mark.parametrize("text", ["between 1m and 2m", "1-2m", "£900k or £1.1m", "a million or two million"])
def test_budget_range_is_ambiguous(text):
    r = parse_budget(text)
    assert r.value is None and r.ambiguous


@pytest.mark.parametrize("text", ["lots", "3", "", "no idea", "three bedrooms", "between one and two million"])
def test_budget_absent(text):
    r = parse_budget(text)
    assert r.value is None and not r.ambiguous


@given(st.integers(min_value=1_000, max_value=500_000_000))
def test_budget_roundtrips_plain_and_comma_numbers(n):
    assert parse_budget(str(n)).value == n
    assert parse_budget(f"£{n:,}").value == n


@pytest.mark.parametrize("text,expected", [
    ("gu5 7aq", "GU5 7AQ"),
    ("GU57AQ", "GU5 7AQ"),
    ("  sw3   3rb ", "SW3 3RB"),
    ("SW10 2FE", "SW10 2FE"),
])
def test_postcode_normalise(text, expected):
    assert normalise_postcode(text) == expected


@pytest.mark.parametrize("text", ["SW3", "hello", "12345", "GU5", "SW3 3R"])
def test_postcode_rejects_partial_or_invalid(text):
    assert normalise_postcode(text) is None


def test_postcode_extracted_from_sentence():
    assert extract_postcode("it's near gu5 7aq, thanks") == "GU5 7AQ"
    assert extract_postcode("no postcode here") is None


def test_phone_validation():
    assert validate_phone("020 7946 0100") == "+442079460100"
    assert validate_phone("my number is 07911 123456") == "+447911123456"
    assert validate_phone("12345") is None
    assert validate_phone("call me maybe") is None


def test_email_validation():
    assert validate_email_address("sam@example.com") == "sam@example.com"
    assert validate_email_address("it's sam@example.com.") == "sam@example.com"
    assert validate_email_address("sam@") is None
    assert validate_email_address("not an email") is None
