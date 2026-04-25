from fastapi import APIRouter, HTTPException, Query

from neural_network.datasets import normalize_sequence
from services.service_state import models

router = APIRouter(prefix="/inference", tags=["Inference"])

from pydantic import BaseModel


class HandsInput(BaseModel):
    hands: list[list[list[list[float]]]]



@router.post("/predict")
async def predict(device_id: str = Query(...), data: HandsInput = None):

    model = models.get(device_id)
    if not model:
        raise HTTPException(404, "Model not loaded")

    results = []

    for hand in data.hands:

        if len(hand) != 20:
            continue

        normalized = normalize_sequence(hand)
        name, conf = model.predict_name(normalized)

        results.append({
            "gesture": name,
            "confidence": conf
        })

    return {"gestures": results}
