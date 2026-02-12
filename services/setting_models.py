from fastapi import APIRouter, HTTPException, Query
from neural_network.neural import Neural
from .inference import models

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/load")
async def load_model(
    device_id: str = Query(...),
    model_name: str = Query(...)
):
    model = Neural(model_name=model_name)

    if not model.load_model():
        raise HTTPException(404, "Model not found")

    models[device_id] = model

    return {
        "status": "ok",
        "device_id": device_id,
        "model": model_name
    }
