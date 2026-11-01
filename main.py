import asyncio
from src.window_tello_refactor.presentation.pipeline import WindowPassApp

if __name__ == "__main__":
    asyncio.run(WindowPassApp().run())