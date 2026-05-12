from fastapi import APIRouter, HTTPException, Query

from neural_network.datasets import normalize_sequence
from services.service_state import models

router = APIRouter(prefix="/inference", tags=["Inference"])

from pydantic import BaseModel


class HandsInput(BaseModel):
    hands: list[list[list[list[float]]]]



@router.post("/predict")
async def predict(device_id: str = Query(...), data: HandsInput = None):

    try:
        model = models.get(device_id)
        if not model:
            raise HTTPException(404, "Model not loaded")

        if not data or not data.hands:
            return {"gestures": []}

        hands = data.hands

        predict_fn = model.predict_name
        normalize = normalize_sequence

        results = []

        for hand in hands:
            if len(hand) != 20:
                continue

            normalized = normalize(hand)
            name, conf = predict_fn(normalized)

            results.append({
                "gesture": name,
                "confidence": conf
            })

        return {"gestures": results}

    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(500, str(e))