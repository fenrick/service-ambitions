from mapping_utils import fit_batch_to_token_cap


def test_fit_batch_to_token_cap_reduces(monkeypatch) -> None:
    """Batch size should shrink until under token cap."""

    logged = {}

    def fake_info(msg: str, *args) -> None:
        logged["msg"] = msg % args if args else msg

    monkeypatch.setattr("mapping_utils.logfire.info", fake_info)

    items = list(range(5))

    def token_counter(seq):
        return len(seq) * 5

    size = fit_batch_to_token_cap(items, 5, 10, token_counter, label="test")

    assert size == 2
    assert "Reduced" in logged["msg"]


def test_fit_batch_to_token_cap_never_below_one() -> None:
    """Helper should always return at least one item."""

    size = fit_batch_to_token_cap([1, 2, 3], 3, 0, lambda s: len(s) * 10, label="test")
    assert size == 1
