import json
import os
import numpy as np


def normalize_sequence(sequence):
    seq = np.asarray(sequence, dtype=np.float32)

    if seq.shape[-1] == 63:
        seq = seq.reshape(len(seq), 21, 3)

    seq = seq - seq[:, 0:1, :]

    max_vals = np.abs(seq).max(axis=(1, 2), keepdims=True)
    max_vals[max_vals == 0] = 1

    seq = seq / max_vals

    return seq.reshape(seq.shape[0], -1).reshape(-1)


class HandDataLoader:

    def __init__(self, dataset_path="asl_dynamic"):
        self.dataset_path = dataset_path
        self.EXPECTED_LEN = 20 * 63
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_abs_path = os.path.join(self.base_path, "dataset", dataset_path)

    def load_data(self):
        if not os.path.exists(self.dataset_abs_path):
            print(f"Ошибка: Путь {self.dataset_abs_path} не найден!")
            return (None, None), (None, None), []

        gestures = sorted([
            name for name in os.listdir(self.dataset_abs_path)
            if os.path.isdir(os.path.join(self.dataset_abs_path, name))
        ])

        x_all = []
        y_all = []
        num_classes = len(gestures)

        for i, gesture_name in enumerate(gestures):
            json_path = os.path.join(
                self.dataset_abs_path,
                gesture_name,
                "results_dynamic.json"
            )

            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            for _, sequence in data.items():
                processed_vector = normalize_sequence(sequence)

                if processed_vector.shape[0] != self.EXPECTED_LEN:
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

        print(f"Загружено: {len(x_all)} динамических примеров ({self.EXPECTED_LEN} входов каждый).")
        print(f"Классы: {gestures}")

        return (
            (x_all[:split_index], y_all[:split_index]),
            (x_all[split_index:], y_all[split_index:]),
            gestures
        )
