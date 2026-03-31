# Sentiment Analysis API

A FastAPI service that classifies text sentiment with Hugging Face transformers.

## Overview

- Single-text and batch prediction endpoints
- Startup model loading with a health endpoint
- Dockerfile included for containerized runs

The current app uses `cardiffnlp/twitter-roberta-base-sentiment-latest` through the Hugging Face `pipeline` API.

## Stack

- Python
- FastAPI
- Hugging Face Transformers
- PyTorch
- Uvicorn
- Docker

## API Endpoints

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | Predict sentiment for one text |
| `POST` | `/predict/batch` | Predict sentiment for up to 32 texts |
| `GET` | `/health` | Check service and model readiness |

## Quick Start

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API docs.

### Docker

```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

Example response:

```json
{
  "text": "This movie was absolutely fantastic!",
  "label": "positive",
  "score": 0.9998,
  "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

## Project Files

```text
sentiment-api/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Notes

- The first startup downloads the model and can take longer than later runs.
- The app is configured for CPU by default. Change `device=-1` to `device=0` in `app.py` if you want GPU inference.
