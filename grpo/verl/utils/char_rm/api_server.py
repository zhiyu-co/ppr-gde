from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from verl.utils.char_rm.model_loader import RewardModelService
import os

# -------- load model once --------
MODEL_PATH = os.getenv("CHAR_RM_MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError("Set CHAR_RM_MODEL_PATH to the local BaichuanCharRM checkpoint path.")
reward_service = RewardModelService(MODEL_PATH)

# -------- FastAPI init --------
app = FastAPI(title="Baichuan Reward Model API")

class Record(BaseModel):
    role_info: str
    context: str
    model_output: str
    metric_zh: str


@app.post("/score")
def score(record: Record):
    """Batch scoring"""
    text = format_input(record)
    score = reward_service.score(text)
    return {"score": score}

@app.post("/score_batch")
def score_batch(records: List[Record]):
    """Batch scoring"""
    texts = [format_input(rec) for rec in records]
    scores = reward_service.score_batch(texts)
    return {"scores": scores}


def format_input(example: Record):
    return f"<RoleInfo>\n\n{example.role_info}\n\n<Context>\n\n{example.context}\n\n<Response>\n\n{example.model_output}\n\n<Dimension>\n\n{example.metric_zh}"
