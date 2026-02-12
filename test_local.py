import os

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from neural_network.datasets import normalize_landmarks
from neural_network.neural import Neural


def main():
    print("Выберите режим:")
    print("[t] - Обучение (Train)")
    print("[r] - Распознавание (Run)")
    mode = input("Введите букву: ").lower()
    name = "asl"
    dataset_path = os.path.join("neural_network\\dataset", name)
    gestures = [n for n in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, n))]
    nn = Neural(input_len=63, output_len=len(gestures), model_name=name)

    if mode == 't':
        print("Загрузка данных и начало обучения...")
        nn.train(epochs=500, batch_size=32, dataset_name=name)
        print("Обучение завершено.")

    elif mode == 'r':
        nn.load_model()

        base_options = python.BaseOptions(model_asset_path="neural_network/models/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            mp_image = Image(image_format=ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            key = cv2.waitKey(1) & 0xFF

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                list_lms = [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]]
                if key == ord(' '):
                    normalized_lms = normalize_landmarks(list_lms)
                    gesture_name, confidence = nn.predict_name(normalized_lms)

                    print(f"Распознано: {gesture_name} ({confidence:.2f})")
                    print(list_lms)
                    cv2.putText(frame, f"SHOT: {gesture_name.upper()}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                for lm in lms:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

            cv2.imshow("Hand Gesture Recognition", frame)

            if key == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Неверный режим. Перезапустите программу.")


if __name__ == "__main__":
    main()
