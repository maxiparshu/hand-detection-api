import os
import random
import shutil
from pathlib import Path
from PIL import Image
from fastapi import APIRouter, HTTPException

from services.service_state import models

router = APIRouter(prefix="/training", tags=["Training"])

GESTURES_DIR = Path("gestures")


@router.get("/random-gesture/{user_id}")
async def get_random_gesture_batch(user_id: str, count: int = 5):
    user_neural = models.get(user_id)
    if not user_neural:
        raise HTTPException(status_code=404, detail="User model not found")

    model_name = user_neural.get_model()
    model_path = GESTURES_DIR / model_name

    if not model_path.exists() or not model_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Model folder not found")

    all_images = [f for f in os.listdir(model_path) if f.lower().endswith(".png")]
    if len(all_images) < (count * 2):
        raise HTTPException(status_code=400, detail="Not enough images for batch")

    user_temp_dir = Path("temp") / user_id
    if user_temp_dir.exists():
        shutil.rmtree(user_temp_dir)
    user_temp_dir.mkdir(parents=True, exist_ok=True)

    batch_result = []

    selected_images = random.sample(all_images, count)

    for img_name in selected_images:
        remaining = [i for i in all_images if i != img_name]
        extra_img = random.choice(remaining)

        src_path = model_path / img_name
        dst_path = user_temp_dir / img_name

        with Image.open(src_path) as image:
            if random.random() < 0.8:
                if random.choice([True, False]):
                    image = image.rotate(random.uniform(-90, 90), expand=True)
                if random.choice([True, False]):
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            image.save(dst_path, format="PNG")

        batch_result.append({
            "model": model_name,
            "gesture_name": Path(img_name).stem,
            "extra_gesture_name": Path(extra_img).stem,
            "image_path": f"/temp/{user_id}/{img_name}"
        })

    return batch_result

@router.get("/all-gestures/{user_id}")
async def get_all_gestures(user_id: str):
    model_name = models.get(user_id).get_model()
    model_path = GESTURES_DIR / model_name

    files = sorted(f for f in os.listdir(model_path) if f.lower().endswith(".png"))

    gestures = [{
        "name": Path(f).stem,
        "url": f"/gestures/{model_name}/{f}"
    } for f in files]

    return {"gestures": gestures}
