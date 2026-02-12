import json
import os

import numpy as np
from sklearn.utils import shuffle


def load_hand_data(dataset_path="hand_dataset"):
    gestures = [name for name in os.listdir(dataset_path)
                if os.path.isdir(os.path.join(dataset_path, name))]

    x_all = []
    y_all = []

    num_classes = len(gestures)

    for i, gesture_name in enumerate(gestures):
        json_path = os.path.join(dataset_path, gesture_name, "results.json")

        if not os.path.exists(json_path):
            print(f"Предупреждение: {json_path} не найден.")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        for img_name, landmarks in data.items():
            flatten_lms = np.array(landmarks).flatten()

            x_all.append(flatten_lms)

            one_hot = np.zeros(num_classes)
            one_hot[i] = 1.0
            y_all.append(one_hot)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    x_all, y_all = shuffle(x_all, y_all, random_state=42)

    split_index = int(len(x_all) * 0.8)

    x_train, x_test = x_all[:split_index], x_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    return (x_train, y_train), (x_test, y_test), gestures


def data_len():
    datasets = "hand_dataset"
    return len([name for name in os.listdir(datasets) if os.path.isdir(os.path.join(datasets, name))])
