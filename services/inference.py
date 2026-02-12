from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from neural_network.datasets import normalize_landmarks
from services.service_state import models

router = APIRouter(prefix="/inference", tags=["Inference"])



class HandsInput(BaseModel):
    hands: list[list[float]]


@router.post("/predict")
async def predict(device_id: str = Query(...), data: HandsInput = None):
    model = models.get(device_id)
    if not model:
        raise HTTPException(404, "Model not loaded for this device")

    results = []

    for raw in data.hands:
        normalized = normalize_landmarks(raw)
        name, conf = model.predict_name(normalized)

        results.append({
            "gesture": name,
            "confidence": conf
        })

    return {"gestures": results}
