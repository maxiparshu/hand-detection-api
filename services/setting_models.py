import json
import os
from fastapi import APIRouter, HTTPException, Query

from neural_network.neural import Neural
from services.service_state import models, CONFIG_FILE

router = APIRouter(prefix="/models", tags=["Models"])


def save_state(device_id, model_name):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)

    state = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                state = json.load(f)
        except json.JSONDecodeError:
            state = {}

    state[device_id] = model_name
    with open(CONFIG_FILE, 'w') as f:
        json.dump(state, f)


@router.post("/load")
async def load_model(
        device_id: str = Query(...),
        model_name: str = Query(...)
):
    model = Neural(model_name=model_name)

    if not model.load_model():
        raise HTTPException(404, "Model not found")

    models[device_id] = model
    save_state(device_id, model_name)

    return {"status": "ok"}