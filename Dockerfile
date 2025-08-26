# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder
WORKDIR /app
COPY pyproject.toml poetry.lock README.md ./
RUN pip install --no-cache-dir poetry && poetry install --only main --no-root
COPY . .
RUN poetry build -f wheel
RUN python -m venv /opt/venv && . /opt/venv/bin/activate && pip install --no-cache-dir dist/*.whl

FROM gcr.io/distroless/python3-debian12:nonroot
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
ENTRYPOINT ["service-ambitions"]
