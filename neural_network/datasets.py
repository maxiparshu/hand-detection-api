import json
import os
import numpy as np


def normalize_landmarks(landmarks):
    lms = np.array(landmarks)

    if lms.shape == (63,):
        lms = lms.reshape(21, 3)

    lms = lms - lms[0]

    max_val = np.max(np.abs(lms))
    if max_val > 0:
        lms = lms / max_val

    return lms.flatten()


def load_hand_data(dataset_path="asl"):
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_abs_path = os.path.join(base_path, "dataset", dataset_path)

    if not os.path.exists(dataset_abs_path):
        print(f"Ошибка: Путь {dataset_abs_path} не найден!")
        return (None, None), (None, None), []

    gestures = sorted([name for name in os.listdir(dataset_abs_path)
                       if os.path.isdir(os.path.join(dataset_abs_path, name))])

    x_all = []
    y_all = []
    num_classes = len(gestures)

    for i, gesture_name in enumerate(gestures):
        json_path = os.path.join(dataset_abs_path, gesture_name, "results.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for img_name, landmarks in data.items():
            processed_lms = normalize_landmarks(landmarks)

            if processed_lms.shape[0] != 63:
                continue

            x_all.append(processed_lms)

            one_hot = np.zeros(num_classes)
            one_hot[i] = 1.0
            y_all.append(one_hot)

    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.float32)

    indices = np.arange(len(x_all))
    np.random.seed(42)
    np.random.shuffle(indices)
    x_all = x_all[indices]
    y_all = y_all[indices]

    split_index = int(len(x_all) * 0.8)

    print(f"Загружено: {len(x_all)} примеров для {num_classes} классов.")

    return (x_all[:split_index], y_all[:split_index]), \
        (x_all[split_index:], y_all[split_index:]), \
        gestures