"""Helpers for enabling LangSmith tracing."""

from __future__ import annotations

import logging
import os

from langsmith import Client

logger = logging.getLogger(__name__)


def init_langsmith(project: str | None = None) -> None:
    """Enable LangSmith tracing if configured.

    Args:
        project: Optional name for the LangSmith project.

    When the ``LANGSMITH_API_KEY`` environment variable is present this
    function activates LangSmith's tracing support by setting the appropriate
    environment variables. If ``project`` is given, traces are grouped under
    that project.
    """

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.debug("LANGSMITH_API_KEY not set; skipping LangSmith setup")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project

    Client()
    logger.info(
        "LangSmith tracing enabled%s",
        f" for project {project}" if project else "",
    )
