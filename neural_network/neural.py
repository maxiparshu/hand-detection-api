import os
from datetime import datetime
import numpy as np

from .activation import ReLU, Softmax
from .datasets import HandDataLoader
from .dense import Dense, Dropout
from .loss import CrossEntropy


class Neural:
    def __init__(self, frames=20, coords=63, output_len=5, reg_lambda=0.01, dropout_p=0.4, learning_rate=0.01,
                 model_name="default_dynamic"):

        self.input_len = frames * coords
        self.output_len = output_len
        self.learning_rate = learning_rate

        h1_len = 256
        h2_len = 128

        self.layer1 = Dense(self.input_len, h1_len, reg_lambda=reg_lambda)
        self.activation1 = ReLU()
        self.dropout1 = Dropout(p=dropout_p)

        self.layer2 = Dense(h1_len, h2_len, reg_lambda=reg_lambda)
        self.activation2 = ReLU()
        self.dropout2 = Dropout(p=dropout_p)

        self.output_layer = Dense(h2_len, output_len, reg_lambda=reg_lambda)
        self.activation_final = Softmax()

        self.loss_function = CrossEntropy()
        self.names = []
        self.model_name = model_name

    def forward(self, x, train=True):
        out = self.layer1.forward(x)
        out = self.activation1.forward(out)
        out = self.dropout1.forward(out, train=train)

        out = self.layer2.forward(out)
        out = self.activation2.forward(out)
        out = self.dropout2.forward(out, train=train)

        out = self.output_layer.forward(out)
        return self.activation_final.forward(out)

    def backward(self, batch_size):
        grad = self.loss_function.backward_batch()
        grad = self.activation_final.backward(grad)

        grad = self.output_layer.backward(grad, self.learning_rate, mini_batch=True, len_mini_batch=batch_size)

        grad = self.dropout2.backward(grad)
        grad = self.activation2.backward(grad)
        grad = self.layer2.backward(grad, self.learning_rate, mini_batch=True, len_mini_batch=batch_size)

        grad = self.dropout1.backward(grad)
        grad = self.activation1.backward(grad)
        self.layer1.backward(grad, self.learning_rate, mini_batch=True, len_mini_batch=batch_size)

    def predict_name(self, sequence_landmarks):
        if not self.names:
            return "Unknown (Model not loaded)", 0.0

        x = np.array(sequence_landmarks).flatten().reshape(1, -1)

        if x.shape[1] != self.input_len:
            return f"Error: Expected {self.input_len} inputs, got {x.shape[1]}", 0.0

        probs = self.forward(x, train=False)
        class_idx = np.argmax(probs)
        confidence = probs[0][class_idx]

        name = self.names[class_idx] if class_idx < len(self.names) else "Unknown"
        return name, confidence * 100

    def train(self, epochs=500, batch_size=32, patience=50):
        dataset_loader = HandDataLoader(self.model_name)
        (x_train, y_train), (x_test, y_test), self.names = dataset_loader.load_data()

        if len(self.names) != self.output_len:
            print(f"Корректировка выходного слоя под {len(self.names)} классов...")
            h2_len = self.layer2.get_weight()[0].shape[1]
            self.output_layer = Dense(h2_len, len(self.names))
            self.output_len = len(self.names)

        print(f"Запуск обучения: {len(x_train)} трен. | {len(x_test)} тест.")

        best_val_loss = float('inf')
        best_weights = {}
        epochs_without_improvement = 0

        for epoch in range(epochs):
            indices = np.arange(len(x_train))
            np.random.shuffle(indices)
            x_shf, y_shf = x_train[indices], y_train[indices]

            for i in range(0, len(x_shf), batch_size):
                x_batch, y_batch = x_shf[i:i + batch_size], y_shf[i:i + batch_size]
                y_pred = self.forward(x_batch, train=True)
                self.loss_function.forward_batch(y_batch, y_pred)
                self.backward(x_batch.shape[0])

            y_val_pred = self.forward(x_test, train=False)
            current_val_loss = self.loss_function.forward_batch(y_test, y_val_pred)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                epochs_without_improvement = 0

                w1, b1 = self.layer1.get_weight()
                w2, b2 = self.layer2.get_weight()
                w3, b3 = self.output_layer.get_weight()

                best_weights = {
                    'W1': w1.copy(), 'b1': b1.copy(),
                    'W2': w2.copy(), 'b2': b2.copy(),
                    'W3': w3.copy(), 'b3': b3.copy()
                }
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 150 == 0:
                self.learning_rate *= 0.5
                print(f"LR снижен до: {self.learning_rate}")

            if (epoch + 1) % 10 == 0:
                acc = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_test, axis=1)) * 100
                print(f"Эпоха {epoch + 1}/{epochs} | Val Loss: {current_val_loss:.5f} | Acc: {acc:.2f}%")

            if epochs_without_improvement >= patience:
                print(f"Остановка: Нет улучшений в течение {patience} эпох.")
                break

        if best_weights:
            print(f"Загрузка весов с наилучшим Val Loss: {best_val_loss:.5f}")
            self.layer1.set_weight(best_weights['W1'], best_weights['b1'])
            self.layer2.set_weight(best_weights['W2'], best_weights['b2'])
            self.output_layer.set_weight(best_weights['W3'], best_weights['b3'])

        self.save_model()

    def save_model(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, self.model_name + ".npz")

        w1, b1 = self.layer1.get_weight()
        w2, b2 = self.layer2.get_weight()
        w3, b3 = self.output_layer.get_weight()

        np.savez(model_path, W1=w1, b1=b1, W2=w2, b2=b2, W3=w3, b3=b3, names=np.array(self.names))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Динамическая модель сохранена: {self.model_name}")

    def load_model(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "models", self.model_name + ".npz")

        if os.path.exists(model_path):
            data = np.load(model_path, allow_pickle=True)
            self.layer1.set_weight(data['W1'], data['b1'])
            self.layer2.set_weight(data['W2'], data['b2'])

            w3, b3 = data['W3'], data['b3']
            self.output_layer = Dense(w3.shape[0], w3.shape[1])
            self.output_layer.set_weight(w3, b3)

            self.names = data['names'].tolist()
            self.output_len = len(self.names)
            print(f"Модель загружена. Классы: {self.names}")
            return True
        return False

    def get_model(self):
        return self.model_name

    def get_names(self):
        return self.names
