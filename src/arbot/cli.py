"""Command-line chat: python -m arbot.cli"""
from arbot.app import build_bot
from arbot.graph.bot import ChatBot


def main() -> None:
    bot, tid = build_bot(), ChatBot.new_thread_id()
    print(f"Bot: {bot.opening()}")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        result = bot.send(tid, text)
        print(f"Bot: {result.reply}")
        if result.ended:
            break


if __name__ == "__main__":
    main()
