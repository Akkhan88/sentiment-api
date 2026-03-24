# Sentiment Analysis API

A lightweight REST API that classifies text as **POSITIVE** or **NEGATIVE** using a fine-tuned DistilBERT transformer model from Hugging Face.

**Stack:** FastAPI · Hugging Face Transformers · PyTorch · Uvicorn

---

## Quick Start (local)

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app:app --reload --port 8000
```

The first launch downloads the model (~260 MB) and caches it locally.

Open the interactive docs at **http://localhost:8000/docs**.

---

## Quick Start (Docker)

```bash
# Build (downloads the model into the image)
docker build -t sentiment-api .

# Run
docker run -p 8000:8000 sentiment-api
```

---

## API Reference

### `POST /predict`

Analyze a single text.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

Response:

```json
{
  "text": "This movie was absolutely fantastic!",
  "label": "POSITIVE",
  "score": 0.9998,
  "model": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

### `POST /predict/batch`

Analyze up to 32 texts in one call.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love it!", "Terrible experience."]}'
```

### `GET /health`

Check service readiness.

```bash
curl http://localhost:8000/health
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port |

To use a **GPU**, change `device=-1` to `device=0` in `app.py`.

---

## Project Structure

```
sentiment-api/
├── app.py              # FastAPI application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Multi-stage Docker build
├── .dockerignore
└── README.md
```
