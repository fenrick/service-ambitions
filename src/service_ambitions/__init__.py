"""Top-level package for service ambition generation."""

from .cli import main
from .generator import ServiceAmbitionGenerator
from .loader import load_prompt, load_services

__all__ = ["main", "load_prompt", "load_services", "ServiceAmbitionGenerator"]
