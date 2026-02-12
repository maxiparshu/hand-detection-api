import os
from datetime import datetime
import numpy as np

from .activation import ReLU, Softmax
from .datasets import load_hand_data
from .dense import Dense, Dropout
from .loss import CrossEntropy


class Neural:
    def __init__(self, input_len=63, output_len=5, reg_lambda=0.01, dropout_p=0.4, learning_rate=0.01, model_name="default"):
        h1_len = 128
        h2_len = 64

        self.output_len = output_len
        self.input_len = input_len
        self.learning_rate = learning_rate

        self.layer1 = Dense(input_len, h1_len, reg_lambda=reg_lambda)
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
        self.log_file = "training_log.txt"

    def forward(self, x, train=True):
        out = self.layer1.forward(x)
        out = self.activation1.forward(out)
        out = self.dropout1.forward(out, train=train)

        out = self.layer2.forward(out)
        out = self.activation2.forward(out)
        out = self.dropout2.forward(out, train=train)

        # Выход
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

    def predict_name(self, landmarks):
        if not self.names:
            return "Unknown (Model not loaded)", 0.0

        x = np.array(landmarks).flatten().reshape(1, -1)

        probs = self.forward(x, train=False)

        class_idx = np.argmax(probs)

        confidence = probs[0][class_idx]

        confidence_pct = confidence * 100

        name = self.names[class_idx] if class_idx < len(self.names) else "Unknown"

        return name, confidence_pct

    def train(self, epochs=500, batch_size=32, dataset_name="hand_dataset"):
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.names = load_hand_data(dataset_name)
        print(f"Запуск обучения: {len(self.x_train)} примеров. Слои: 128 -> 64")

        for epoch in range(epochs):
            indices = np.arange(len(self.x_train))
            np.random.shuffle(indices)
            x_shf, y_shf = self.x_train[indices], self.y_train[indices]
            total_loss = 0

            for i in range(0, len(x_shf), batch_size):
                x_batch, y_batch = x_shf[i:i + batch_size], y_shf[i:i + batch_size]
                y_pred = self.forward(x_batch, train=True)
                total_loss += self.loss_function.forward_batch(y_batch, y_pred)
                self.backward(x_batch.shape[0])

            if (epoch + 1) % 100 == 0:
                self.learning_rate *= 0.5
                print(f"Epoch {epoch + 1}: LR reduced to {self.learning_rate}")
            if (epoch + 1) % 10 == 0:
                avg_l = total_loss / (len(x_shf) / batch_size)
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_l:.5f}")

        self.save_model()

    def save_model(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "models", self.model_name + ".npz")

        # Получаем веса из всех трех слоев
        w1, b1 = self.layer1.get_weight()
        w2, b2 = self.layer2.get_weight()
        w3, b3 = self.output_layer.get_weight()

        np.savez(model_path, W1=w1, b1=b1, W2=w2, b2=b2, W3=w3, b3=b3, names=np.array(self.names))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Модель и имена ({self.names}) сохранены {self.model_name}")

    def load_model(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "models", self.model_name + ".npz")

        print(model_path)

        if os.path.exists(model_path):
            data = np.load(model_path, allow_pickle=True)
            self.layer1.set_weight(data['W1'], data['b1'])
            self.layer2.set_weight(data['W2'], data['b2'])
            self.output_layer.set_weight(data['W3'], data['b3'])
            self.names = data['names'].tolist()
            self.output_len = len(self.names)
            print(f"Модель загружена успешно. Имена: ({self.names})")
            return True
        return False

    def get_model(self):
        return self.model_name
