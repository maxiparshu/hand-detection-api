import os
from collections import deque

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from neural_network.datasets import normalize_sequence
from neural_network.generate_dynamic import DynamicDatasetGenerator
from neural_network.neural import Neural


def ask_model_name():
    while True:
        name = input("Введите название модели (например: asl): ").strip().lower()

        if name.endswith("_dynamic"):
            name = name.replace("_dynamic", "")

        return name, f"{name}_dynamic"


def main():
    print("=== Система обучения жестам (Динамический режим) ===")
    print("[t] - Обучение (Train )")
    print("[r] - Распознавание (Run)")
    mode = input("Выберите режим: ").lower()

    name, model_name = ask_model_name()
    FRAMES_WINDOW = 20
    MODEL_PATH = "neural_network/models/hand_landmarker.task"

    dataset_path = os.path.join("neural_network", "dataset", model_name)
    print(dataset_path)
    gestures = []
    if os.path.exists(dataset_path):
        gestures = [n for n in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, n))]

    nn = Neural(frames=FRAMES_WINDOW, output_len=len(gestures), model_name=model_name)

    if mode == 't':
        mode = input("Новый датасет(y): ").lower()
        if mode == 'y':
            generator = DynamicDatasetGenerator(model_name=name,
                                                dataset_path=dataset_path,
                                                gestures_folder="gestures_original//",
                                                model_asset_path=MODEL_PATH)
            generator.generate_dataset()
        print(f"\nНачало обучения на датасете: {model_name}")
        nn.train(epochs=500, batch_size=32)
        print("Обучение завершено. Модель сохранена.")

    elif mode == 'r':
        if not nn.load_model():
            print("Ошибка: Модель не найдена. Сначала проведите обучение [t].")
            return

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        detector = vision.HandLandmarker.create_from_options(options)

        sequence_buffer = deque(maxlen=FRAMES_WINDOW)

        cap = cv2.VideoCapture(0)
        current_prediction = "Waiting..."
        confidence = 0.0

        print("\nРаспознавание запущено. Нажмите 'q' для выхода.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                raw_coords = [[lm.x, lm.y, lm.z] for lm in lms]

                sequence_buffer.append(raw_coords)

                for lm in lms:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

                if len(sequence_buffer) == FRAMES_WINDOW:
                    processed_sequence = normalize_sequence(list(sequence_buffer))

                    gesture_name, conf = nn.predict_name(processed_sequence)

                    if conf > 75:
                        current_prediction = gesture_name
                        confidence = conf
                    else:
                        current_prediction = "Uncertain..."
                        confidence = conf
            else:
                sequence_buffer.clear()
                current_prediction = "No hand"
                confidence = 0.0

            color = (0, 255, 0) if confidence > 75 else (0, 0, 255)

            cv2.putText(frame, f"GESTURE: {current_prediction.upper()}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, f"CONF: {confidence:.1f}%", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.rectangle(frame, (10, 110), (10 + len(sequence_buffer) * 15, 125), (255, 100, 0), -1)
            cv2.putText(frame, "BUFFER", (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            cv2.imshow("Sign Language AI Tutor (Dynamic)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Ошибка: выбран неверный режим.")


if __name__ == "__main__":
    main()
