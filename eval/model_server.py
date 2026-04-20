"""
OpenAI-compatible API server wrapping the fine-tuned Qwen2.5-Coder LoRA model.
Uses Unsloth 4-bit loading to fit in 8GB VRAM.

Usage:
    python eval/model_server.py [--port 8000] [--model-path ./code-review-model/lora]

Exposes:
    GET  /v1/models           — list available models
    POST /v1/chat/completions — chat completions (OpenAI format)
    GET  /health              — health check
"""

import argparse
import time
import uuid
import warnings

warnings.filterwarnings("ignore")

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

# ── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--model-path", type=str, default="./code-review-model/lora")
parser.add_argument("--max-seq-len", type=int, default=2048)
SERVER_ARGS = parser.parse_args()

MODEL_NAME = "qwen2.5-coder-7b-finetuned"

# ── Load model ───────────────────────────────────────────────────────────────

print(f"Loading model from {SERVER_ARGS.model_path} ...")
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SERVER_ARGS.model_path,
    max_seq_length=SERVER_ARGS.max_seq_len,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model loaded successfully.")

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Qwen Code Review Model Server")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = Field(default=2048, alias="max_tokens")
    max_completion_tokens: Optional[int] = None
    response_format: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids.to("cuda")
    else:
        input_ids = inputs.to("cuda")

    prompt_len = input_ids.shape[1]

    max_new = min(request.max_completion_tokens or request.max_tokens or 2048, 2048)
    temp = request.temperature if request.temperature and request.temperature > 0 else 0.01
    top_p = request.top_p or 0.9

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            temperature=temp,
            top_p=top_p,
            do_sample=True,
            use_cache=False,
        )

    generated = output[0][prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    completion_tokens = len(generated)
    prompt_tokens = prompt_len

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=SERVER_ARGS.port)
