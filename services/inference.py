from fastapi import FastAPI, HTTPException
from fastapi.params import Query
from pydantic import BaseModel
from neural_network.generate import normalize_landmarks
from neural_network.neural import Neural

app = FastAPI()

models: dict[str, Neural] = {}

class HandsInput(BaseModel):
    hands: list[list[float]]  # список рук, каждая рука — список координат


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z



@app.post("/load_model")
async def load_model(
    device_id: str = Query(..., description="Unique device id"),
    model_name: str = Query(..., description="Model filename")
):
    model = Neural(model_name=model_name)

    if not model.load_model():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    models[device_id] = model

    return {
        "status": "ok",
        "device_id": device_id,
        "model": model_name
    }



@app.post("/predict")
async def predict_gestures(
    device_id: str = Query(...),
    data: HandsInput = None
):
    model = models.get(device_id)

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not loaded for this device"
        )

    results = []

    try:
        for raw_coords in data.hands:
            points = [
                Point(raw_coords[i], raw_coords[i + 1], raw_coords[i + 2])
                for i in range(0, len(raw_coords), 3)
            ]

            normalized_data = normalize_landmarks(points)
            name, confidence_pct = model.predict_name(normalized_data)

            results.append({
                "gesture": str(name),
                "confidence": float(confidence_pct)
            })

        return {"gestures": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
