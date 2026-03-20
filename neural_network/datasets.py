import json
import os

import numpy as np


def normalize_sequence(sequence):
    normalized_seq = []
    for landmarks in sequence:
        lms = np.array(landmarks)
        if lms.shape == (63,):
            lms = lms.reshape(21, 3)

        lms = lms - lms[0]

        max_val = np.max(np.abs(lms))
        if max_val > 0:
            lms = lms / max_val

        normalized_seq.append(lms.flatten())

    return np.array(normalized_seq).flatten()


def load_hand_data(dataset_path="asl_dynamic"):
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

    EXPECTED_LEN = 20 * 63

    for i, gesture_name in enumerate(gestures):
        json_path = os.path.join(dataset_abs_path, gesture_name, "results_dynamic.json")
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for sample_name, sequence in data.items():
            processed_vector = normalize_sequence(sequence)

            if processed_vector.shape[0] != EXPECTED_LEN:
                continue

            x_all.append(processed_vector)

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

    print(f"Загружено: {len(x_all)} динамических примеров ({EXPECTED_LEN} входов каждый).")
    print(f"Классы: {gestures}")

    return (x_all[:split_index], y_all[:split_index]), \
        (x_all[split_index:], y_all[split_index:]), \
        gestures
