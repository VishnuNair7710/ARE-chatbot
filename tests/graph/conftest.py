import pytest

from arbot.data.loaders import load_listings, load_seller_coverage
from arbot.graph.bot import ChatBot


_LISTINGS, _COVERAGE = load_listings(), load_seller_coverage()   # load the CSVs once per run


class Chat:
    """Drives a conversation: say() sends a line and returns the bot's reply."""

    def __init__(self, **kw):
        kw.setdefault("listings", _LISTINGS)
        kw.setdefault("coverage", _COVERAGE)
        self.bot = ChatBot(**kw)
        self.tid = self.bot.new_thread_id()
        self.last = None

    def say(self, *lines: str) -> str:
        for line in lines:
            self.last = self.bot.send(self.tid, line)
        return self.last.reply

    @property
    def state(self):
        return self.last.state

    def to_buy_question(self):
        """Hi -> name -> buy; leaves the bot at 'Do you have a specific property in mind?'"""
        self.say("hi", "Sam", "I want to buy")
        return self


@pytest.fixture
def chat():
    return Chat()


@pytest.fixture(scope="session")
def homes():
    """One unambiguous listing per (category, status), found by name through the real index."""
    from arbot.data.property_index import PropertyIndex
    df = load_listings()
    idx = PropertyIndex(df)
    out = {}
    for cat in ("new_build", "resale"):
        for status in ("available", "under_offer", "sold"):
            for r in df[(df.listing_category == cat) & (df.status == status)].itertuples():
                if (len(r.property_name.split()) >= 2
                        and [h.property_id for h in idx.find(r.property_name)] == [r.property_id]):
                    out[(cat, status)] = r
                    break
    return out


@pytest.fixture(scope="session")
def data():
    df = load_listings()
    nb, rs = df[df.listing_category == "new_build"], df[df.listing_category == "resale"]

    def pcs(frame, pred):
        return sorted({pc for pc, g in frame.groupby("postcode") if pred(g)})

    avail = lambda g: (g.status == "available").any()
    none_avail = lambda g: not avail(g)
    return dict(
        nb_ok=pcs(nb, avail)[0],
        nb_listed_none_available=pcs(nb, none_avail)[0],
        nb_absent=sorted(set(df.postcode) - set(nb.postcode))[0],
        rs_ok=pcs(rs, avail)[0],
        rs_listed_none_available=pcs(rs, none_avail)[0],
        rs_absent=sorted(set(df.postcode) - set(rs.postcode))[0],
        covered=sorted(load_seller_coverage())[0],
        uncovered=sorted(set(df.postcode) - load_seller_coverage())[0],
    )
