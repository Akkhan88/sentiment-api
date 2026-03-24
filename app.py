"""
Sentiment Analysis API
======================
A FastAPI service powered by a Hugging Face transformer model
(distilbert-base-uncased-finetuned-sst-2-english).

Endpoints:
  POST /predict        — Analyze sentiment of a single text
  POST /predict/batch  — Analyze sentiment of multiple texts
  GET  /health         — Health check
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline


# ---------------------------------------------------------------------------
# Global model holder
# ---------------------------------------------------------------------------
class ModelState:
    classifier = None
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    load_time: float = 0.0


model_state = ModelState()


# ---------------------------------------------------------------------------
# Lifespan – load the model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model: {model_state.model_name} ...")
    start = time.time()
    model_state.classifier = pipeline(
        "sentiment-analysis",
        model=model_state.model_name,
        device=-1,  # CPU; change to 0 for GPU
    )
    model_state.load_time = round(time.time() - start, 2)
    print(f"Model loaded in {model_state.load_time}s")
    yield
    print("Shutting down …")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify text as POSITIVE or NEGATIVE using a DistilBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, examples=["I love this product!"])


class BatchInput(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=32)


class SentimentResult(BaseModel):
    text: str
    label: str
    score: float
    model: str


class BatchResult(BaseModel):
    results: list[SentimentResult]
    count: int


class HealthResponse(BaseModel):
    status: str
    model: str
    model_load_time_sec: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify(text: str) -> SentimentResult:
    if model_state.classifier is None:
        raise HTTPException(503, detail="Model not loaded yet")
    result = model_state.classifier(text, truncation=True, max_length=512)[0]
    return SentimentResult(
        text=text,
        label=result["label"],
        score=round(result["score"], 4),
        model=model_state.model_name,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=SentimentResult)
async def predict(body: TextInput):
    """Analyze the sentiment of a single piece of text."""
    return _classify(body.text)


@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(body: BatchInput):
    """Analyze sentiment for up to 32 texts in one call."""
    results = [_classify(t) for t in body.texts]
    return BatchResult(results=results, count=len(results))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check whether the service and model are ready."""
    return HealthResponse(
        status="ok" if model_state.classifier else "loading",
        model=model_state.model_name,
        model_load_time_sec=model_state.load_time,
    )
