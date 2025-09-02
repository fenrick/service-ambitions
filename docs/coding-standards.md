# Coding Standards

This project follows the principles outlined in *Clean Code: A Handbook of Agile Software Craftsmanship*.

## Core Principles

- **Single Purpose**: Each module, class and function should do one thing and do it well with a clear, descriptive name.
- **Functions on Objects**: Prefer methods that operate on object state instead of free functions whenever object context exists.
- **Singleton Shared State**: Shared state should be encapsulated behind a single well‑defined interface to minimise global impact.

## Function Arguments

The ideal number of arguments for a function is zero (niladic). Next comes one (monadic), followed closely by two (dyadic). Three arguments (triadic) should be avoided where possible. More than three (polyadic) requires very special justification—and then shouldn’t be used anyway.

## Formatting and Style

Code must conform to PEP 8 and is automatically formatted with [Black](https://black.readthedocs.io/) and checked with [Ruff](https://docs.astral.sh/ruff/).
