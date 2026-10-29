"""Entry point."""

import asyncio
import os

from config.config import AppConfig
from presentation.app import WindowPassApp


async def _run() -> None:
    cfg = AppConfig()
    app = WindowPassApp(cfg)
    await app.run()


def main() -> None:
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_run())


if __name__ == "__main__":
    main()