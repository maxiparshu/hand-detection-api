import json
import os
import random

import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

GESTURES = ['like', 'dislike', 'stop', 'ok', 'peace', 'rock',
            '1', '2', '3', '4', '5']
SAMPLES_PER_GESTURE = 50
ANGLES = [-15, 15, -20, 20, 10]
PADDING = 25
DATASET_PATH = "../hand_dataset"
NOISE_LEVEL = 0.008


def rotate_image(image, angle):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))


def normalize_landmarks(landmarks):
    lms = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    base = lms[0]
    translated = lms - base
    max_dist = np.max(np.linalg.norm(translated, axis=1))
    if max_dist > 0:
        normalized = translated / max_dist
    else:
        normalized = translated
    return normalized.tolist()


def mirror_landmarks(lms_list):
    return [[-lm[0], lm[1], lm[2]] for lm in lms_list]


def add_noise(lms_list, level):
    return [[lm[0] + random.uniform(-level, level),
             lm[1] + random.uniform(-level, level),
             lm[2] + random.uniform(-level, level)] for lm in lms_list]



if __name__ == "__main__":
    base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7
    )
    detector = vision.HandLandmarker.create_from_options(options)

    for g in GESTURES:
        os.makedirs(os.path.join(DATASET_PATH, g), exist_ok=True)

    cap = cv2.VideoCapture(0)

    for gesture in GESTURES:
        print(f"\nЗАПИСЬ: {gesture.upper()} (S - снять, Space - пропустить жест, Q - выход)")
        count = 0
        gesture_data = {}
        skip_gesture = False

        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            if not ret: break

            display_frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                xs = [int(lm.x * w) for lm in lms]
                ys = [int(lm.y * h) for lm in lms]
                x_min, y_min = max(0, min(xs) - PADDING), max(0, min(ys) - PADDING)
                x_max, y_max = min(w, max(xs) + PADDING), min(h, max(ys) + PADDING)
                cv2.rectangle(display_frame, (w - x_max, y_min), (w - x_min, y_max), (0, 255, 0), 2)

                key = cv2.waitKey(1)

                if key & 0xFF == ord('s'):
                    norm_lms = normalize_landmarks(lms)
                    base_name = f"{gesture}_{count}"

                    crop = frame[y_min:y_max, x_min:x_max]
                    if crop.size > 0:
                        crop_resized = cv2.resize(crop, (224, 224))
                        cv2.imwrite(os.path.join(DATASET_PATH, gesture, f"{base_name}_orig.jpg"), crop_resized)
                    gesture_data[f"{base_name}_orig.jpg"] = norm_lms

                    for ang in ANGLES:
                        key_name = f"{base_name}_ang{ang}.jpg"
                        gesture_data[key_name] = norm_lms
                        if count == 0:
                            rot_img = rotate_image(frame, ang)
                            r_crop = rot_img[y_min:y_max, x_min:x_max]
                            if r_crop.size > 0:
                                cv2.imwrite(os.path.join(DATASET_PATH, gesture, key_name),
                                            cv2.resize(r_crop, (224, 224)))

                    gesture_data[f"{base_name}_mir.jpg"] = mirror_landmarks(norm_lms)
                    if count == 0 and crop.size > 0:
                        cv2.imwrite(os.path.join(DATASET_PATH, gesture, f"{base_name}_mir.jpg"),
                                    cv2.flip(crop_resized, 1))

                    gesture_data[f"{base_name}_noise.jpg"] = add_noise(norm_lms, NOISE_LEVEL)

                    count += 1
                    print(f"Записано: {count}/{SAMPLES_PER_GESTURE}", end="\r")

                elif key & 0xFF == ord(' '):
                    print(f"\nЖест {gesture} пропущен пользователем.")
                    skip_gesture = True
                    break

                elif key & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            cv2.putText(display_frame, f"REC: {gesture} | {count}/{SAMPLES_PER_GESTURE}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Data Collector", display_frame)

            if not result.hand_landmarks:
                key = cv2.waitKey(1)
                if key & 0xFF == ord(' '):
                    skip_gesture = True
                    break
                elif key & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        if gesture_data:
            with open(os.path.join(DATASET_PATH, gesture, "results.json"), "w") as f:
                json.dump(gesture_data, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()
