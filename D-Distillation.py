import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, datasets
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Split CIFAR-10 training data into multiple devices (e.g., 4 devices)
def split_data(x, y, num_devices):
    split_size = len(x) // num_devices
    return [(x[i * split_size:(i + 1) * split_size], y[i * split_size:(i + 1) * split_size]) for i in range(num_devices)]

num_devices = 4
device_data = split_data(x_train, y_train, num_devices)

# Shared reference dataset (a subset of CIFAR-10)
reference_size = 1000
x_ref, y_ref = x_train[:reference_size], y_train[:reference_size]

# Define a simple CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Initialize models for each device
devices = [create_model() for _ in range(num_devices)]

# Compile models
for device in devices:
    device.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Distributed Distillation Training Loop
class DistributedDistillation:
    def __init__(self, devices, reference_data, num_classes=10, alpha=0.5, beta=1.0):
        self.devices = devices
        self.reference_data = reference_data
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def train(self, device_data, epochs=10, batch_size=64):
        x_ref, _ = self.reference_data
        ref_logits = [np.zeros((len(x_ref), self.num_classes)) for _ in range(len(self.devices))]

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")

            # Train each device on its private data
            for i, (model, (x_dev, y_dev)) in enumerate(zip(self.devices, device_data)):
                model.fit(x_dev, y_dev, epochs=1, batch_size=batch_size, verbose=0)

                # Generate soft decisions (logits) on the reference data
                ref_logits[i] = model.predict(x_ref, batch_size=batch_size)

            # Aggregate soft decisions (average over all devices)
            aggregated_logits = np.mean(ref_logits, axis=0)

            # Distillation step: Train each device to match the aggregated logits
            for i, model in enumerate(self.devices):
                with tf.GradientTape() as tape:
                    # Compute device predictions on the reference data
                    device_logits = model(x_ref, training=True)

                    # Distillation loss: Match aggregated logits
                    distillation_loss = tf.reduce_mean(
                        tf.keras.losses.categorical_crossentropy(
                            tf.nn.softmax(aggregated_logits / self.beta),
                            tf.nn.softmax(device_logits / self.beta)
                        )
                    )

                    # Combine with supervised loss (if needed, based on private data)
                    total_loss = self.alpha * distillation_loss

                # Apply gradients
                grads = tape.gradient(total_loss, model.trainable_weights)
                model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Evaluate each device
            for i, (model, (x_dev, y_dev)) in enumerate(zip(self.devices, device_data)):
                loss, acc = model.evaluate(x_dev, y_dev, verbose=0)
                print(f"  Device {i + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

# Initialize and train
reference_data = (x_ref, y_ref)
dd = DistributedDistillation(devices, reference_data, alpha=0.5, beta=1.0)
dd.train(device_data, epochs=10, batch_size=64)

# Evaluate final performance
for i, model in enumerate(devices):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Evaluation - Device {i + 1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
