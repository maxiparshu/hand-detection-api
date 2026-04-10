from neural_network.neural import Neural
import json

with open("settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

MODELS_DIR = settings["MODELS_DIR"]
MODEL_NAMES_MAP = settings["MODEL_NAMES_MAP"]
CONFIG_FILE = "temp/loaded_models.json",


models: dict[str, Neural] = {}
