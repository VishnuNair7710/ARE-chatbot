import pandas as pd
import pytest

from arbot.data.loaders import CATEGORIES, STATUSES, load_listings, load_seller_coverage
from arbot.domain.postcode import normalise_postcode
from arbot.domain.qualify import qualify_new_build, qualify_resale, qualify_seller


def _row(i, category, postcode, price, status="available", beds=2):
    return dict(property_id=f"T-{i:03d}", property_name=f"Home {i}", listing_category=category,
                postcode=postcode, price_gbp=price, status=status, bedrooms=beds)


@pytest.fixture
def listings():
    rows = [
        _row(1, "new_build", "AA1 1AA", 1_200_000),
        _row(2, "new_build", "AA1 1AA", 1_500_000),
        _row(3, "new_build", "BB1 1BB", 1_300_000, status="sold"),
        _row(4, "new_build", "CC1 1CC", 1_100_000, status="under_offer"),
        _row(5, "resale", "DD1 1DD", 500_000),
        _row(6, "resale", "EE1 1EE", 900_000, status="sold"),
        _row(7, "resale", "FF1 1FF", 400_000, status="under_offer"),
    ]
    return pd.DataFrame(rows)


# ---- Lane A -------------------------------------------------------------
@pytest.mark.parametrize("budget,kind", [
    (999_999, "M1"), (0, "M1"),
    (1_000_000, "M3"),  # passes the £1M gate (inclusive); cheapest listing there is £1.2M
    (1_200_000, "QUALIFIED"),
])
def test_new_build_budget_threshold(listings, budget, kind):
    assert qualify_new_build(budget, "AA1 1AA", listings).kind == kind


def test_m1_takes_precedence_over_unknown_postcode(listings):
    assert qualify_new_build(500_000, "ZZ9 9ZZ", listings).kind == "M1"


def test_new_build_no_listing_at_postcode_is_m2(listings):
    assert qualify_new_build(2_000_000, "ZZ9 9ZZ", listings).kind == "M2"
    # a resale-only postcode has no *new-build* listing
    assert qualify_new_build(2_000_000, "DD1 1DD", listings).kind == "M2"


@pytest.mark.parametrize("postcode", ["BB1 1BB", "CC1 1CC"])
def test_new_build_listed_but_not_available_is_m3(listings, postcode):
    assert qualify_new_build(2_000_000, postcode, listings).kind == "M3"


def test_new_build_available_but_over_budget_is_m3(listings):
    assert qualify_new_build(1_100_000, "AA1 1AA", listings).kind == "M3"


def test_new_build_budget_is_inclusive_and_filters(listings):
    out = qualify_new_build(1_200_000, "AA1 1AA", listings)
    assert out.qualified and [m.price for m in out.matches] == [1_200_000]


def test_matches_capped_at_five_cheapest_first():
    rows = [_row(i, "new_build", "AA1 1AA", 2_000_000 - i * 1000) for i in range(8)]
    out = qualify_new_build(3_000_000, "AA1 1AA", pd.DataFrame(rows))
    prices = [m.price for m in out.matches]
    assert len(prices) == 5 and prices == sorted(prices)


# ---- Lane B -------------------------------------------------------------
def test_resale_ignores_budget_and_qualifies(listings):
    out = qualify_resale("DD1 1DD", listings)
    assert out.qualified and out.matches[0].property_id == "T-005"


def test_resale_m2_and_m3(listings):
    assert qualify_resale("ZZ9 9ZZ", listings).kind == "M2"
    assert qualify_resale("AA1 1AA", listings).kind == "M2"  # only new-build there
    assert qualify_resale("EE1 1EE", listings).kind == "M3"  # sold
    assert qualify_resale("FF1 1FF", listings).kind == "M3"  # under offer is not available


# ---- Lane D -------------------------------------------------------------
def test_seller_coverage():
    assert qualify_seller("GU5 7AQ", {"GU5 7AQ"}).qualified
    assert qualify_seller("GU5 7AP", {"GU5 7AQ"}).kind == "M2"


# ---- Real data ----------------------------------------------------------
def test_real_seller_coverage_loaded():
    cov = load_seller_coverage()
    assert len(cov) == 100 and "GU5 7AQ" in cov
    assert qualify_seller(normalise_postcode("gu5 7aq"), cov).qualified


def test_real_data_integrity():
    df = load_listings()
    assert len(df) == 247 and df["property_id"].is_unique
    assert set(df["status"]) <= STATUSES and set(df["listing_category"]) <= CATEGORIES
    assert df["postcode"].map(lambda p: normalise_postcode(p) == p).all()
    # business assumption behind M1: the agency lists no new-build under £1M
    assert (df.loc[df["listing_category"] == "new_build", "price_gbp"] >= 1_000_000).all()


def test_real_data_every_new_build_postcode_qualifies_at_high_budget_if_any_available():
    df = load_listings()
    for pc, grp in df[df["listing_category"] == "new_build"].groupby("postcode"):
        out = qualify_new_build(50_000_000, pc, df)
        has_avail = (grp["status"] == "available").any()
        assert out.qualified == has_avail
        assert all(m.price <= 50_000_000 for m in out.matches) and len(out.matches) <= 5
