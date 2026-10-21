"""Values the scenarios refer to as ${name}. They are derived from the real CSVs, so the eval
keeps working if the data changes (instead of hard-coding postcodes that may stop qualifying)."""
from __future__ import annotations

from arbot.data.loaders import load_listings, load_seller_coverage
from arbot.data.property_index import PropertyIndex


def variables() -> dict[str, str]:
    df = load_listings()
    coverage = load_seller_coverage()
    idx = PropertyIndex(df)
    nb, rs = df[df.listing_category == "new_build"], df[df.listing_category == "resale"]

    def postcodes(frame, pred):
        return sorted({pc for pc, g in frame.groupby("postcode") if pred(g)})

    avail = lambda g: (g.status == "available").any()
    v = {
        "nb_ok": postcodes(nb, avail)[0],
        "nb_none_available": postcodes(nb, lambda g: not avail(g))[0],
        "nb_absent": sorted(set(df.postcode) - set(nb.postcode))[0],
        "rs_ok": postcodes(rs, avail)[0],
        "rs_none_available": postcodes(rs, lambda g: not avail(g))[0],
        "rs_absent": sorted(set(df.postcode) - set(rs.postcode))[0],
        "covered": sorted(coverage)[0],
        "uncovered": sorted(set(df.postcode) - coverage)[0],
    }
    v["covered_compact_lower"] = v["covered"].replace(" ", "").lower()
    for cat, short in (("new_build", "nb"), ("resale", "rs")):
        for status in ("available", "under_offer", "sold"):
            for r in df[(df.listing_category == cat) & (df.status == status)].itertuples():
                if (len(r.property_name.split()) >= 2
                        and [h.property_id for h in idx.find(r.property_name)] == [r.property_id]):
                    v[f"{short}_{status}_name"] = r.property_name
                    v[f"{short}_{status}_beds"] = str(r.bedrooms)
                    v[f"{short}_{status}_price"] = f"£{r.price_gbp:,}"
                    break
    return v
