import asyncio

from token_scheduler import TokenScheduler


async def _run_scheduler(completed: list[str]) -> None:
    scheduler = TokenScheduler(max_workers=2)

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


async def _collect_results() -> list[str]:
    scheduler = TokenScheduler(max_workers=2)

    async def job(duration: float, label: str) -> str:
        await asyncio.sleep(duration)
        return label

    # Submit tasks with tokens out of order
    scheduler.submit(lambda: job(0.03, "first"), tokens=30)
    scheduler.submit(lambda: job(0.01, "second"), tokens=10)
    scheduler.submit(lambda: job(0.02, "third"), tokens=20)

    return await scheduler.run()


def test_results_return_in_submission_order() -> None:
    results = asyncio.run(_collect_results())
    assert results == ["first", "second", "third"]
