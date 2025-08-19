import asyncio

import pytest

from token_scheduler import TokenScheduler


async def _run_scheduler(completed: list[str]) -> None:
    scheduler = TokenScheduler(max_workers=2, context_window=100)

    async def job(duration: float, label: str) -> str:
        await asyncio.sleep(duration)
        completed.append(label)
        return label

    scheduler.submit(lambda: job(0.03, "long"), tokens=30)
    scheduler.submit(lambda: job(0.01, "short"), tokens=10)
    scheduler.submit(lambda: job(0.02, "medium"), tokens=20)

    await scheduler.run()


def test_shorter_tasks_complete_first() -> None:
    completed: list[str] = []
    asyncio.run(_run_scheduler(completed))
    assert completed == ["short", "medium", "long"]


def test_submit_over_context_window_raises() -> None:
    scheduler = TokenScheduler(max_workers=1, context_window=50)

    with pytest.raises(ValueError):
        scheduler.submit(lambda: asyncio.sleep(0), tokens=60)
