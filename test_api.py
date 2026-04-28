import random

from locust import HttpUser, task, between


def generate_hand():
    return [
        [[random.random(), random.random(), random.random()] for _ in range(21)]
        for _ in range(20)
    ]


class HandDetectionUser(HttpUser):
    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.device_id = "test_device"

    @task(5)
    def predict(self):
        payload = {
            "hands": [generate_hand() for _ in range(random.randint(1, 3))]
        }

        self.client.post(
            f"/inference/predict?device_id={self.device_id}",
            json=payload
        )

    @task(1)
    def get_models(self):
        self.client.get("/models/names")

    @task(1)
    def gestures(self):
        self.client.get(f"/training/all-gestures/{self.device_id}")
