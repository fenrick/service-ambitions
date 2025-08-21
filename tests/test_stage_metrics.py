from stage_metrics import (
    iter_stage_totals,
    log_stage_totals,
    record_stage_metrics,
    reset_stage_totals,
)


def test_record_stage_metrics_tracks_prompt_tokens() -> None:
    """Prompt token estimates should accumulate per stage."""

    reset_stage_totals()
    record_stage_metrics("alpha", 10, 1.0, 2.0, False, 4)
    stage, totals = next(iter_stage_totals())
    assert stage == "alpha"
    assert totals.total_tokens == 10
    assert totals.prompt_tokens_estimate == 4
    assert totals.estimated_cost == 1.0
    assert totals.total_duration == 2.0
    assert totals.prompts == 1
    assert totals.errors_429 == 0


def test_log_stage_totals_emits_totals(monkeypatch) -> None:
    """Aggregated metrics should be emitted via logfire."""

    reset_stage_totals()
    record_stage_metrics("alpha", 10, 1.0, 2.0, False, 4)
    logs: list[tuple[str, dict[str, object]]] = []

    def capture(msg: str, **kwargs: object) -> None:
        logs.append((msg, kwargs))

    monkeypatch.setattr("stage_metrics.logfire.info", capture)
    log_stage_totals()

    assert logs
    msg, data = logs[0]
    assert msg == "Stage totals"
    assert data["stage"] == "alpha"
    assert data["total_tokens"] == 10
