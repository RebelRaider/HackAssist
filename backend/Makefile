up:
	poetry run uvicorn app:app --reload --port 8000

.PHONY: migrate-revision
migrate-rev:
	poetry run alembic revision --autogenerate -m $(name)

.PHONY: migrate-up
migrate-up:
	poetry run alembic upgrade $(rev)

.PHONY: local
local:
	docker compose -f docker-compose.local.yml up

.PHONY: test
test:
	poetry run pytest
