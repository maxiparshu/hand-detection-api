from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


def main():
    INPUT_DIR = Path("gestures")
    BACKUP_DIR = Path("gestures_original")

    MODEL_PATH = "real-esrgan-x4plus-128.onnx"
    TARGET_SIZE = 256

    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    for src_path in tqdm(BACKUP_DIR.rglob("*.png"), desc="Upscaling"):
        try:
            # относительный путь внутри gestures_original
            rel = src_path.relative_to(BACKUP_DIR)

            # путь сохранения в gestures
            dst_path = INPUT_DIR / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # --- загрузка ---
            img = Image.open(src_path).convert("RGB")
            img = img.resize((128, 128), Image.LANCZOS)

            # --- в модель ---
            img_np = np.asarray(img).astype(np.float32) / 255.0
            img_np = img_np.transpose(2, 0, 1)
            img_np = img_np[None, ...]

            out = sess.run(
                [output_name],
                {input_name: img_np}
            )[0]

            # --- из модели ---
            out = out.squeeze(0).transpose(1, 2, 0)
            out = (out * 255.0).clip(0, 255).astype(np.uint8)
            out_img = Image.fromarray(out, "RGB")

            out_img = out_img.resize(
                (TARGET_SIZE, TARGET_SIZE),
                Image.LANCZOS
            )

            out_img.save(dst_path, "PNG", optimize=True)

        except Exception as e:
            print(f"Ошибка {src_path}: {e}")


if __name__ == "__main__":
    main()
