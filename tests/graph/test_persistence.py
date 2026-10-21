from arbot.app import build_checkpointer
from arbot.config import Settings
from arbot.data.loaders import load_listings, load_seller_coverage
from arbot.graph import prompts as P
from arbot.graph.bot import ChatBot


def new_bot(db):
    return ChatBot(listings=load_listings(), coverage=load_seller_coverage(),
                   checkpointer=build_checkpointer(Settings(checkpoint_db=db)))


def test_sessions_survive_a_restart_with_the_sqlite_checkpointer(tmp_path):
    db = str(tmp_path / "sessions.db")
    first = new_bot(db)
    tid = first.new_thread_id()
    first.send(tid, "hi")
    first.send(tid, "Sam")
    restarted = new_bot(db)                       # a brand-new process would do exactly this
    assert restarted.has_thread(tid)
    assert restarted.send(tid, "sell").reply == P.POSTCODE_SELLER
    assert restarted.send(tid, "gu5 7aq").state["qualified"]


def test_default_is_in_memory():
    assert build_checkpointer(Settings(checkpoint_db="")) is None
