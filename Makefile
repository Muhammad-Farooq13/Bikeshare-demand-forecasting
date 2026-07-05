.PHONY: install ingest features train explain test api docker-build docker-run lint clean

install:
	pip install -r requirements.txt

ingest:
	python -m src.ingest

features: ingest
	python -m src.features

train: ingest
	python -m src.train

explain: train
	python -m src.explain

test:
	pytest --cov=src --cov=api --cov-report=term-missing

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -f docker/Dockerfile -t bikeshare-demand-api:latest .

docker-run:
	docker compose up --build

lint:
	python -m py_compile src/*.py api/*.py tests/*.py

clean:
	rm -rf data/hour_raw.csv data/hour_features.csv models/*.joblib artifacts/*.json artifacts/*.png .pytest_cache __pycache__ */__pycache__
