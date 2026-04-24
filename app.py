import base64
import io
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Literal

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig


#MODEL_ID = "huihui-ai/Huihui-Qwen3.5-4B-abliterated"
MODEL_ID = "huihui-ai/Huihui-Qwen3.5-2B-abliterated"
#MODEL_ID = "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"
#MODEL_ID = "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated"
#MODEL_ID = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
PUBLIC_MODEL_NAME = "local-qwen"

USE_4BIT = True
DEFAULT_TEMPERATURE = 0.5
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_MAX_NEW_TOKENS = 2000
MAX_IMAGE_SIZE = 896

if "Qwen3-" in MODEL_ID:
    ModelLoader = Qwen3VLForConditionalGeneration
else:
    ModelLoader = AutoModelForImageTextToText


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        type_: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        self.status_code = status_code
        self.message = message
        self.type_ = type_
        self.param = param
        self.code = code


class ImageURL(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    response_format: dict[str, Any] | None = None


def preprocess_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(MAX_IMAGE_SIZE / max(w, h), 1.0)
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)))
    return image


def load_image_from_url(url: str) -> Image.Image:
    if url.startswith("data:image/"):
        try:
            _, b64_data = url.split(",", 1)
            raw = base64.b64decode(b64_data)
        except Exception as e:
            raise ApiError(400, f"Invalid data URL image: {e}", param="messages")
    elif url.startswith("http://") or url.startswith("https://"):
        try:
            with urllib.request.urlopen(url) as resp:
                raw = resp.read()
        except Exception as e:
            raise ApiError(400, f"Could not download image URL: {e}", param="messages")
    else:
        raise ApiError(
            400,
            "Only http(s) URLs and data URLs are supported for image_url.url in this POC.",
            param="messages",
        )

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ApiError(400, f"Could not decode image: {e}", param="messages")

    return preprocess_image(image)


def to_qwen_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    qwen_messages: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg.content, str):
            qwen_messages.append(
                {
                    "role": msg.role,
                    "content": [
                        {
                            "type": "text",
                            "text": msg.content,
                        }
                    ],
                }
            )
            continue

        content_parts: list[dict[str, Any]] = []
        for part in msg.content:
            if part.type == "text":
                content_parts.append(
                    {
                        "type": "text",
                        "text": part.text or "",
                    }
                )
            elif part.type == "image_url":
                if msg.role != "user":
                    raise ApiError(
                        400,
                        "Images are only supported in user messages in this POC.",
                        param="messages",
                    )
                if not part.image_url or not part.image_url.url:
                    raise ApiError(
                        400,
                        "image_url content part is missing image_url.url",
                        param="messages",
                    )
                image = load_image_from_url(part.image_url.url)
                content_parts.append(
                    {
                        "type": "image",
                        "image": image,
                    }
                )
            else:
                raise ApiError(
                    400,
                    f"Unsupported content part type: {part.type}",
                    param="messages",
                )

        qwen_messages.append(
            {
                "role": msg.role,
                "content": content_parts,
            }
        )

    return qwen_messages


def load_model_and_processor():
    bnb_config = None
    dtype = torch.float16

    if USE_4BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        dtype = None

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model_kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "trust_remote_code": True,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = dtype or torch.float16

    model = ModelLoader.from_pretrained(
        MODEL_ID,
        **model_kwargs,
    )
    model.eval()

    return model, processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, processor = load_model_and_processor()
    app.state.model = model
    app.state.processor = processor
    app.state.generation_lock = threading.Lock()
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yield
    # Optional cleanup on shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.type_,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/v1/models")
def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": PUBLIC_MODEL_NAME,
                "object": "model",
                "created": now,
                "owned_by": "local",
            },
            {
                "id": MODEL_ID,
                "object": "model",
                "created": now,
                "owned_by": "local",
            },
        ],
    }


def run_inference(req: ChatCompletionRequest) -> dict[str, Any]:
    if req.response_format is not None:
        raise ApiError(
            501,
            "response_format is reserved in this API but not implemented for the local Qwen backend yet.",
            type_="not_implemented_error",
            param="response_format",
            code="not_supported",
        )

    if req.stream:
        raise ApiError(
            501,
            "stream=true is not implemented in this minimal POC.",
            type_="not_implemented_error",
            param="stream",
            code="not_supported",
        )

    if req.model and req.model not in {PUBLIC_MODEL_NAME, MODEL_ID}:
        raise ApiError(
            400,
            f"Unknown model '{req.model}'. Use '{PUBLIC_MODEL_NAME}' or '{MODEL_ID}'.",
            param="model",
            code="model_not_found",
        )

    temperature = DEFAULT_TEMPERATURE if req.temperature is None else req.temperature
    max_new_tokens = (
        req.max_completion_tokens
        or req.max_tokens
        or DEFAULT_MAX_NEW_TOKENS
    )
    do_sample = temperature > 0

    qwen_messages = to_qwen_messages(req.messages)

    processor = app.state.processor
    model = app.state.model

    inputs = processor.apply_chat_template(
        qwen_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(app.state.device)
    inputs.pop("token_type_ids", None)

    with app.state.generation_lock:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=(temperature if do_sample else 0.0),
                repetition_penalty=DEFAULT_REPETITION_PENALTY,
            )

    prompt_tokens = int(inputs["input_ids"].shape[1])
    completion_tokens = int(generated_ids.shape[1] - prompt_tokens)

    trimmed_ids = generated_ids[:, prompt_tokens:]
    output_text = processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    now = int(time.time())
    return {
        "id": f"chatcmpl-local-{now}",
        "object": "chat.completion",
        "created": now,
        "model": req.model or PUBLIC_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    return run_inference(req)