#!/usr/bin/env python3
"""One-click local CPU inference server for Duan.

Duan is a fine-tuned Laya System-1 decision engine for emergency and clinical
decision tasks, released on Hugging Face (weights only). This script loads the
checkpoint once on CPU and serves a small HTTP API. Training is not supported.

Model source
    Default: ``wipen/Duan`` (Hugging Face repo id). Override it with the
    ``DUAN_MODEL`` environment variable or the ``--model`` flag; a local
    directory (e.g. one fetched by ``huggingface-cli download``) works the same
    way. The weights are not bundled in this repository -- on first run Laya
    downloads only ``model.safetensors``, ``rl_agent_config.json``,
    ``tokenizer/*`` and ``encoder/*``. If huggingface.co is unreachable from
    your network, set ``HF_ENDPOINT`` (e.g. https://hf-mirror.com).

Start
    python serve.py                       # 127.0.0.1:8000
    python serve.py --model <id> --port 8080
    DUAN_MODEL=<id> python serve.py

    Open http://127.0.0.1:8000 in a browser for the bundled web demo
    (Playground + example use cases). The API stays at /predict, /health, /docs.

Dependencies (system Python, no virtual environment required)
    pip install -r requirements.txt
    # requirements.txt declares laya[serve], which pulls torch, transformers,
    # safetensors, huggingface_hub, fastapi and uvicorn.

Entry point
    ``python serve.py`` -> builds the agent -> runs the FastAPI app via uvicorn.

API
    GET  /health    {"status": "ok", "model": "...", "device": "cpu"}
    POST /predict   {"state": {...}, "questions": {...}} -> Laya result dict
    GET  /docs      interactive OpenAPI docs (FastAPI)
"""

import argparse
import os
from pathlib import Path

import laya
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

DEFAULT_MODEL = os.environ.get("DUAN_MODEL", "wipen/Duan")


def build_agent(model_id: str) -> laya.Agent:
    """Load the checkpoint, pinned to CPU regardless of local accelerators."""
    return laya.Agent(model_id, device="cpu")


def make_app(agent: laya.Agent) -> FastAPI:
    """Build the FastAPI app around a pre-loaded agent."""
    app = FastAPI(title="duan-serve", summary="Duan System-1 decisions over HTTP")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": agent.model_id, "device": str(agent.device)}

    @app.post("/predict")
    def predict(body: dict) -> dict:
        questions = body.get("questions")
        if not isinstance(questions, dict):
            raise HTTPException(
                status_code=400,
                detail="body must be {'state': ..., 'questions': {...}}",
            )
        try:
            # Synchronous CPU inference; FastAPI runs this on a worker thread.
            return agent.predict(state=body.get("state"), questions=questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Serve the bundled web demo (demo/index.html) at "/", after the API routes.
    demo_dir = Path(__file__).resolve().parent / "demo"
    if demo_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(demo_dir), html=True), name="demo")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Duan on local CPU.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HF repo id or local dir (default: %(default)s)")
    parser.add_argument("--host", default=os.environ.get("DUAN_HOST", "127.0.0.1"),
                        help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DUAN_PORT", "8000")),
                        help="bind port (default: %(default)s)")
    args = parser.parse_args()

    uvicorn.run(make_app(build_agent(args.model)),
                host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
