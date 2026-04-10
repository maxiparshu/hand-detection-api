from fastapi import APIRouter, HTTPException, Query

from neural_network.datasets import normalize_sequence
from services.service_state import models

router = APIRouter(prefix="/inference", tags=["Inference"])

from pydantic import BaseModel


class HandsInput(BaseModel):
    hands: list[list[list[float]]]


@router.post("/predict")
async def predict(device_id: str = Query(...), data: HandsInput = None):
    model = models.get(device_id)
    if not model:
        raise HTTPException(404, "Model not loaded")

    if len(data.hands) != 20:
        raise HTTPException(400, f"Need 20 frames, got {len(data.hands)}")
    normalized = normalize_sequence(data.hands)

    name, conf = model.predict_name(normalized)

    return {
        "gestures": [
            {
                "gesture": name,
                "confidence": conf
            }
        ]
    }
