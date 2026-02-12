import json
import os
import random
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SAMPLES_PER_GESTURE = 50
ANGLES = [-15, 15, -20, 20, 10]
PADDING = 25
NOISE_RANGE = (0.003, 0.018)

DATASET_PATH = "dataset/asl"
ASL_FOLDER = "../gestures_original/asl"


def rotate_landmarks_math(lms_list, angle_deg):
    angle_rad = np.radians(-angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotated = []
    for x, y, z in lms_list:
        nx = (x - 0.5) * c - (y - 0.5) * s + 0.5
        ny = (x - 0.5) * s + (y - 0.5) * c + 0.5
        rotated.append([nx, ny, z])
    return rotated


def add_random_noise(lms_list, intensity_range):
    current_level = random.uniform(*intensity_range)

    return [[x + random.uniform(-current_level, current_level),
             y + random.uniform(-current_level, current_level),
             z + random.uniform(-current_level, current_level)] for x, y, z in lms_list]

if __name__ == "__main__":
    os.makedirs(DATASET_PATH, exist_ok=True)

    if not os.path.exists(ASL_FOLDER):
        print(f"Ошибка: Папка {ASL_FOLDER} не найдена!")
        exit()

    image_files = [f for f in os.listdir(ASL_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)

    for file_name in image_files:
        gesture = os.path.splitext(file_name)[0]
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        sample_img = cv2.imread(os.path.join(ASL_FOLDER, file_name))

        count = 0
        gesture_data = {}

        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret: break

            h, w = frame.shape[:2]
            display_frame = cv2.flip(frame, 1)

            mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                xs = [int(lm.x * w) for lm in lms]
                ys = [int(lm.y * h) for lm in lms]
                cv2.rectangle(display_frame, (w - max(xs) - PADDING, min(ys) - PADDING),
                              (w - min(xs) + PADDING, max(ys) + PADDING), (0, 255, 0), 2)

            sample_h = h
            sample_w = int(sample_img.shape[1] * (sample_h / sample_img.shape[0]))
            resized_sample = cv2.resize(sample_img, (sample_w, sample_h))

            cv2.putText(display_frame, f"GESTURE: {gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display_frame, f"COUNT: {count}/{SAMPLES_PER_GESTURE}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
            cv2.putText(resized_sample, "ETALON", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            combined_view = np.hstack((display_frame, resized_sample))
            cv2.imshow("Data Collector", combined_view)

            key = cv2.waitKey(1)
            if key == ord('s') and result.hand_landmarks:
                raw_lms = [[lm.x, lm.y, lm.z] for lm in lms]
                base_name = f"{gesture}_{count}"

                orig_img_name = f"{base_name}_orig.jpg"
                cv2.imwrite(os.path.join(gesture_dir, orig_img_name), frame)
                gesture_data[orig_img_name] = raw_lms

                for ang in ANGLES:
                    key_name = f"{base_name}_ang{ang}.jpg"
                    gesture_data[key_name] = rotate_landmarks_math(raw_lms, ang)
                    if count == 0:
                        M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
                        cv2.imwrite(os.path.join(gesture_dir, key_name), cv2.warpAffine(frame, M, (w, h)))

                mir_key = f"{base_name}_mir.jpg"
                gesture_data[mir_key] = [[1.0 - x, y, z] for x, y, z in raw_lms]
                if count == 0:
                    cv2.imwrite(os.path.join(gesture_dir, mir_key), cv2.flip(frame, 1))

                for i in range(5):
                    noise_key = f"{base_name}_noise_{i}.jpg"
                    gesture_data[noise_key] = add_random_noise(raw_lms, NOISE_RANGE)
                count += 1

            elif key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if gesture_data:
            with open(os.path.join(gesture_dir, "results.json"), "w") as f:
                json.dump(gesture_data, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()