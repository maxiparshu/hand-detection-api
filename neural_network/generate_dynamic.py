import json
import os
import random
from pathlib import Path
from PIL import Image as ImagePIL, ImageDraw, ImageFont
import PIL
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def normalize_sequence(lms_list):
    wrist = lms_list[0]
    return [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in lms_list]


def ask_model_name():
    while True:
        folder = input("Введите папку: ").strip()
        name = input("Введите название модели (например: asl): ").strip()

        base = Path(__file__).resolve().parent
        folder_path = (base.parent / folder / name).resolve()

        if os.path.exists(folder_path):
            return name, folder_path
        else:
            print(f"Папка '{folder_path}' не найдена. Попробуйте снова.\n")


def rotate_landmarks_math(lms_list, angle_deg):
    angle_rad = np.radians(-angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    rotated = []
    for x, y, z in lms_list:
        nx = (x - 0.5) * c - (y - 0.5) * s + 0.5
        ny = (x - 0.5) * s + (y - 0.5) * c + 0.5
        rotated.append([nx, ny, z])

    return rotated


def rotate_sequence(sequence, angle_deg):
    return [rotate_landmarks_math(lms, angle_deg) for lms in sequence]


class DynamicDatasetGenerator:

    def __init__(self, model_name, gestures_folder,
                 dataset_path="dataset/",
                 model_asset_path="models/hand_landmarker.task"):
        self.SAMPLES_PER_GESTURE = 50
        self.FRAMES_PER_SEQUENCE = 20
        self.ANGLES = [-15, 15, -20, 20, 10]
        self.PADDING = 25
        self.NOISE_RANGE = (0.003, 0.018)

        self.model_name = model_name

        self.dataset_path = dataset_path
        self.gestures_folder = gestures_folder

        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)

    def add_noise_to_sequence(self, sequence):
        return [self.add_random_noise(lms) for lms in sequence]

    def add_random_noise(self, lms_list):
        current_level = random.uniform(*self.NOISE_RANGE)

        return [
            [
                x + random.uniform(-current_level, current_level),
                y + random.uniform(-current_level, current_level),
                z + random.uniform(-current_level, current_level)
            ]
            for x, y, z in lms_list
        ]

    def generate_dataset(self):
        global display_frame
        os.makedirs(self.dataset_path, exist_ok=True)

        if not os.path.exists(self.gestures_folder):
            print(f"Ошибка: Папка {self.gestures_folder} не найдена!")
            return

        image_files = [f for f in os.listdir(self.gestures_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for file_name in image_files:
            gesture = os.path.splitext(file_name)[0]
            gesture_dir = os.path.join(self.dataset_path, gesture)
            os.makedirs(gesture_dir, exist_ok=True)
            path = os.path.join(self.gestures_folder, file_name)

            sample_img = cv2.imdecode(
                np.fromfile(path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            count = 0
            gesture_data = {}

            font = PIL.ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 32)
            while count < self.SAMPLES_PER_GESTURE:

                ret, frame = self.cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                display_frame = cv2.flip(frame, 1)

                mp_image = Image(
                    image_format=ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                result = self.detector.detect(mp_image)

                if result.hand_landmarks:
                    lms = result.hand_landmarks[0]
                    xs = [int(lm.x * w) for lm in lms]
                    ys = [int(lm.y * h) for lm in lms]

                    cv2.rectangle(
                        display_frame,
                        (w - max(xs) - self.PADDING, min(ys) - self.PADDING),
                        (w - min(xs) + self.PADDING, max(ys) + self.PADDING),
                        (0, 255, 0), 2
                    )

                sample_h = h
                sample_w = int(sample_img.shape[1] * (sample_h / sample_img.shape[0]))
                resized_sample = cv2.resize(sample_img, (sample_w, sample_h))

                img_pil = ImagePIL.fromarray(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                )

                draw = PIL.ImageDraw.Draw(img_pil)

                text = f"GESTURE: {gesture}"

                draw.text(
                    (10, 10),
                    text,
                    font=font,
                    fill=(255, 0, 0)
                )

                display_frame = cv2.cvtColor(
                    np.array(img_pil),
                    cv2.COLOR_RGB2BGR
                )

                cv2.putText(
                    display_frame,
                    f"COUNT: {count}/{self.SAMPLES_PER_GESTURE}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.putText(display_frame, f"COUNT: {count}/{self.SAMPLES_PER_GESTURE}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(display_frame, "Press 'S' to RECORD SEQUENCE", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                combined_view = np.hstack((display_frame, resized_sample))
                cv2.imshow("Dynamic Data Collector", combined_view)

                key = cv2.waitKey(1)

                if key == ord('s'):
                    sequence = []
                    print(f"Recording {gesture}...")

                    while len(sequence) < self.FRAMES_PER_SEQUENCE:
                        ret, inner_frame = self.cap.read()
                        if not ret:
                            break

                        inner_mp_image = Image(
                            image_format=ImageFormat.SRGB,
                            data=cv2.cvtColor(inner_frame, cv2.COLOR_BGR2RGB)
                        )
                        inner_result = self.detector.detect(inner_mp_image)

                        if inner_result.hand_landmarks:
                            raw_lms = [[lm.x, lm.y, lm.z] for lm in inner_result.hand_landmarks[0]]
                            sequence.append(normalize_sequence(raw_lms))

                        cv2.imshow("Dynamic Data Collector", cv2.flip(inner_frame, 1))
                        cv2.waitKey(1)

                    if len(sequence) == self.FRAMES_PER_SEQUENCE:
                        base_name = f"{gesture}_{count}"

                        gesture_data[f"{base_name}_orig"] = sequence

                        for ang in self.ANGLES:
                            gesture_data[f"{base_name}_ang{ang}"] = rotate_sequence(sequence, ang)

                        gesture_data[f"{base_name}_mir"] = [
                            [[-x, y, z] for x, y, z in lms] for lms in sequence
                        ]

                        for i in range(3):
                            gesture_data[f"{base_name}_noise_{i}"] = self.add_noise_to_sequence(sequence)

                        count += 1

                elif key == ord('q'):
                    self.cleanup()
                    return
                elif key == ord(' '):
                    gesture_data = {}
                    break
            if gesture_data:
                with open(os.path.join(gesture_dir, "results_dynamic.json"), "w") as f:
                    json.dump(gesture_data, f, indent=4)

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    name, folder_path = ask_model_name()
    generator = DynamicDatasetGenerator(name, gestures_folder=folder_path)
    generator.generate_dataset()
