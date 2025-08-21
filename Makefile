.PHONY: test image

TEST ?= poetry run pytest

test:
	$(TEST)

image:
	docker build -t service-ambitions:latest .
