"""Entry point."""

import asyncio
import os

from app import WindowPassApp
from config import AppConfig


async def _run() -> None:
    cfg = AppConfig()
    app = WindowPassApp(cfg)
    await app.run()


def main() -> None:
    # On Windows, SelectorEventLoop is a better fit for socket-heavy code,
    # but Proactor (the default) also works fine with to_thread. Use
    # SelectorEventLoop only if you hit odd socket/task issues.
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())


if __name__ == "__main__":
    main()