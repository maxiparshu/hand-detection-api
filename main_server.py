import json
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from neural_network.neural import Neural
from services import (inference, setting_models, training)
from services.service_state import models, CONFIG_FILE

app = FastAPI(title="Hand Detection API")


@app.on_event("startup")
async def startup_event():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                state = json.load(f)

            for device_id, model_name in state.items():
                model = Neural(model_name=model_name)
                if model.load_model():
                    models[device_id] = model
                    print(f"[*] Восстановлена модель для {device_id}: {model_name}")
        except Exception as e:
            print(f"[!] Ошибка восстановления: {e}")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене лучше заменить на конкретный домен
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hand Detection API is running!", "docs": "/docs"}

app.include_router(inference.router)
app.include_router(setting_models.router)
app.include_router(training.router)

app.mount("/gestures", StaticFiles(directory="gestures"), name="gestures")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")
