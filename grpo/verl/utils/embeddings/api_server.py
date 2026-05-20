import torch
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


# -------- load model once --------
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError("Set EMBEDDING_MODEL_PATH to the local embedding model path or HF model id.")
reward_service = SentenceTransformer(MODEL_PATH)

# -------- FastAPI init --------
app = FastAPI(title="Baichuan Reward Model API")


def to_serializable(x):
    """将数组 / Tensor / numpy 类型递归转换为可 JSON 序列化的 python 类型。"""

    # torch.Tensor -> list
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()

    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()

    # numpy scalar -> Python scalar
    if isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
        return x.item()

    # dict[str, ...]
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}

    # list / tuple
    if isinstance(x, (list, tuple)):
        return [to_serializable(v) for v in x]

    # already OK
    return x


class EncodeBatchRequest(BaseModel):
    texts: List[str]
    batch_size: int = 64
    show_progress_bar: bool = False
    normalize_embeddings: bool = False


@app.post("/encode_batch")
def encode_batch(req: EncodeBatchRequest):
    """
    使用 reward_service.encode(texts, batch_size, show_progress_bar)
    返回 [N, 768] embedding
    """

    embeddings = reward_service.encode(
        req.texts,
        batch_size=req.batch_size,
        show_progress_bar=req.show_progress_bar,
        normalize_embeddings=req.normalize_embeddings
    )

    embeddings = to_serializable(embeddings)

    return {"embeddings": embeddings}
