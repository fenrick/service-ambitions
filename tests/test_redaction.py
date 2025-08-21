"""Tests for the :mod:`redaction` helpers."""

from redaction import redact_pii


def test_redact_pii_masks_common_patterns() -> None:
    """Emails, identifiers and digits should be redacted."""

    text = "Email j.doe@example.com ID AB1234567 number 1234"
    assert redact_pii(text) == "Email <redacted> ID <redacted> number ****"
