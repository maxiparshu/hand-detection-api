import json
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from neural_network.neural import Neural
from services import (inference, setting_models, training)
from services.service_state import models, CONFIG_FILE


class HandDetectionApp:
    def __init__(self):
        self.app = FastAPI(title="Hand Detection API")
        self.config_file = CONFIG_FILE
        self.models_registry = models

        self._setup_middleware()
        self._setup_routers()
        self._setup_static()
        self._register_events()

    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routers(self):
        self.app.include_router(inference.router)
        self.app.include_router(setting_models.router)
        self.app.include_router(training.router)

    def _setup_static(self):
        self.app.mount("/gestures", StaticFiles(directory="gestures"), name="gestures")
        self.app.mount("/temp", StaticFiles(directory="temp"), name="temp")

    def _register_events(self):
        @self.app.on_event("startup")
        async def startup():
            await self.on_startup()

        @self.app.get("/")
        async def root():
            return {"message": "Hand Detection API is running!", "docs": "/docs"}

    async def on_startup(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    state = json.load(f)

                for device_id, model_name in state.items():
                    model = Neural(model_name=model_name)
                    if model.load_model():
                        self.models_registry[device_id] = model
                        print(f"Восстановлена модель для {device_id}: {model_name}")
            except Exception as e:
                print(f"Ошибка восстановления: {e}")


server = HandDetectionApp()
app = server.app
