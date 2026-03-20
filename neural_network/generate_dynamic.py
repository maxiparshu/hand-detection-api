import json
import os
import random
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SAMPLES_PER_GESTURE = 50
FRAMES_PER_SEQUENCE = 20
ANGLES = [-15, 15, -20, 20, 10]
PADDING = 25
NOISE_RANGE = (0.003, 0.018)

DATASET_PATH = "dataset/asl_dynamic"
ASL_FOLDER = "../gestures_original/asl"


def rotate_sequence(sequence, angle_deg):
    return [rotate_landmarks_math(lms, angle_deg) for lms in sequence]


def rotate_landmarks_math(lms_list, angle_deg):
    angle_rad = np.radians(-angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotated = []
    for x, y, z in lms_list:
        nx = (x - 0.5) * c - (y - 0.5) * s + 0.5
        ny = (x - 0.5) * s + (y - 0.5) * c + 0.5
        rotated.append([nx, ny, z])
    return rotated


def add_noise_to_sequence(sequence, intensity_range):
    return [add_random_noise(lms, intensity_range) for lms in sequence]


def add_random_noise(lms_list, intensity_range):
    current_level = random.uniform(*intensity_range)
    return [[x + random.uniform(-current_level, current_level),
             y + random.uniform(-current_level, current_level),
             z + random.uniform(-current_level, current_level)] for x, y, z in lms_list]


def normalize_sequence(lms_list):
    wrist = lms_list[0]
    return [[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in lms_list]

def generate_dataset():
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
            cv2.putText(display_frame, "Press 'S' to RECORD SEQUENCE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

            combined_view = np.hstack((display_frame, resized_sample))
            cv2.imshow("Dynamic Data Collector", combined_view)

            key = cv2.waitKey(1)

            if key == ord('s'):
                sequence = []
                print(f"Recording {gesture}...")

                while len(sequence) < FRAMES_PER_SEQUENCE:
                    ret, inner_frame = cap.read()
                    if not ret: break

                    inner_mp_image = Image(image_format=ImageFormat.SRGB,
                                           data=cv2.cvtColor(inner_frame, cv2.COLOR_BGR2RGB))
                    inner_result = detector.detect(inner_mp_image)

                    if inner_result.hand_landmarks:
                        raw_lms = [[lm.x, lm.y, lm.z] for lm in inner_result.hand_landmarks[0]]
                        sequence.append(normalize_sequence(raw_lms))

                        progress = int((len(sequence) / FRAMES_PER_SEQUENCE) * 100)
                        cv2.putText(inner_frame, f"RECORDING: {progress}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

                    cv2.imshow("Dynamic Data Collector", cv2.flip(inner_frame, 1))
                    cv2.waitKey(1)

                if len(sequence) == FRAMES_PER_SEQUENCE:
                    base_name = f"{gesture}_{count}"

                    gesture_data[f"{base_name}_orig"] = sequence

                    for ang in ANGLES:
                        gesture_data[f"{base_name}_ang{ang}"] = rotate_sequence(sequence, ang)

                    gesture_data[f"{base_name}_mir"] = [[[-x, y, z] for x, y, z in lms] for lms in sequence]

                    for i in range(3):
                        gesture_data[f"{base_name}_noise_{i}"] = add_noise_to_sequence(sequence, NOISE_RANGE)

                    count += 1

            elif key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        if gesture_data:
            with open(os.path.join(gesture_dir, "results_dynamic.json"), "w") as f:
                json.dump(gesture_data, f, indent=4)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_dataset()