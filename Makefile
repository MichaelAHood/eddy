test:
	poetry run pytest

format:
	poetry run ruff --fix .


.PHONY: test