import json
import os
from fastapi import APIRouter, HTTPException, Query

from neural_network.neural import Neural
from services.service_state import models, CONFIG_FILE, MODELS_DIR, MODEL_NAMES_MAP

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



@router.get("/names")
async def get_available_models():
    if not os.path.exists(MODELS_DIR):
        raise HTTPException(status_code=404, detail="Directory not found")

    raw_files = [
        f.removesuffix(".npz")
        for f in os.listdir(MODELS_DIR)
        if os.path.isfile(os.path.join(MODELS_DIR, f)) and f.endswith(".npz")
    ]

    models_list = [
        {
            "id": name,
            "name": MODEL_NAMES_MAP.get(name, name)
        }
        for name in raw_files
    ]

    return {"models": models_list}


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
