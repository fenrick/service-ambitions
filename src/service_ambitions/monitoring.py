"""Helpers for enabling LangSmith tracing."""

from __future__ import annotations

import logging
import os

from langsmith import Client

logger = logging.getLogger(__name__)


def init_langsmith(project: str | None = None, api_key: str | None = None) -> None:
    """Enable LangSmith tracing if configured.

    Args:
        project: Optional name for the LangSmith project.
        api_key: LangSmith API key. Falls back to ``LANGSMITH_API_KEY`` env var.

    When the API key is available this function activates LangSmith's tracing
    support by setting the appropriate environment variables. If ``project`` is
    given, traces are grouped under that project.
    """

    key = api_key or os.getenv("LANGSMITH_API_KEY")
    if not key:
        logger.debug("LANGSMITH_API_KEY not set; skipping LangSmith setup")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project

    Client()
    logger.info(
        "LangSmith tracing enabled%s",
        f" for project {project}" if project else "",
    )
