try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses, datasets, optimizers
    import numpy as np
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise ImportError("TensorFlow is not installed. Please ensure TensorFlow is installed in your environment.") from e

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

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

# Define the KL divergence loss for codistillation
def kl_divergence_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log((y_true + 1e-10) / (y_pred + 1e-10)), axis=-1))

# Codistillation training loop
class CoDistillation:
    def __init__(self, model_1, model_2, temperature=3.0, alpha=0.5):
        self.model_1 = model_1
        self.model_2 = model_2
        self.temperature = temperature
        self.alpha = alpha

    def train(self, x_train, y_train, x_test, y_test, epochs=20, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        optimizer_1 = optimizers.Adam()
        optimizer_2 = optimizers.Adam()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for step, (x_batch, y_batch) in enumerate(dataset):
                # One-hot encode labels
                y_batch_onehot = tf.one_hot(tf.squeeze(y_batch), depth=10)

                with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
                    # Forward pass for both models
                    logits_1 = self.model_1(x_batch, training=True)
                    logits_2 = self.model_2(x_batch, training=True)

                    # Compute soft predictions
                    soft_logits_1 = tf.nn.softmax(logits_1 / self.temperature)
                    soft_logits_2 = tf.nn.softmax(logits_2 / self.temperature)

                    # Supervised loss
                    loss_1 = losses.categorical_crossentropy(y_batch_onehot, logits_1)
                    loss_2 = losses.categorical_crossentropy(y_batch_onehot, logits_2)

                    # Codistillation loss
                    distillation_loss_1 = kl_divergence_loss(soft_logits_2, soft_logits_1)
                    distillation_loss_2 = kl_divergence_loss(soft_logits_1, soft_logits_2)

                    # Combine supervised and codistillation losses
                    total_loss_1 = self.alpha * loss_1 + (1 - self.alpha) * distillation_loss_1
                    total_loss_2 = self.alpha * loss_2 + (1 - self.alpha) * distillation_loss_2

                # Compute gradients
                grads_1 = tape_1.gradient(total_loss_1, self.model_1.trainable_weights)
                grads_2 = tape_2.gradient(total_loss_2, self.model_2.trainable_weights)

                # Apply gradients
                optimizer_1.apply_gradients(zip(grads_1, self.model_1.trainable_weights))
                optimizer_2.apply_gradients(zip(grads_2, self.model_2.trainable_weights))

            # Evaluate both models
            acc_1 = self.evaluate(self.model_1, x_test, y_test)
            acc_2 = self.evaluate(self.model_2, x_test, y_test)
            print(f"Model 1 Accuracy: {acc_1:.4f}, Model 2 Accuracy: {acc_2:.4f}")

    @staticmethod
    def evaluate(model, x_test, y_test):
        results = model.evaluate(x_test, y_test, verbose=0)
        return results[1]

# Knowledge Distillation
class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.5):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        optimizer = optimizers.Adam()

        # Train teacher model
        print("Training Teacher Model...")
        self.teacher_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.teacher_model.fit(x_train, y_train, epochs=5, batch_size=batch_size, verbose=1)

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for step, (x_batch, y_batch) in enumerate(dataset):
                y_batch_onehot = tf.one_hot(tf.squeeze(y_batch), depth=10)

                with tf.GradientTape() as tape:
                    # Get teacher predictions
                    teacher_logits = self.teacher_model(x_batch, training=False)
                    teacher_soft_logits = tf.nn.softmax(teacher_logits / self.temperature)

                    # Get student predictions
                    student_logits = self.student_model(x_batch, training=True)
                    student_soft_logits = tf.nn.softmax(student_logits / self.temperature)

                    # Supervised loss
                    supervised_loss = losses.categorical_crossentropy(y_batch_onehot, student_logits)

                    # Distillation loss
                    distillation_loss = kl_divergence_loss(teacher_soft_logits, student_soft_logits)

                    # Combined loss
                    total_loss = self.alpha * supervised_loss + (1 - self.alpha) * distillation_loss

                # Compute gradients
                grads = tape.gradient(total_loss, self.student_model.trainable_weights)

                # Apply gradients
                optimizer.apply_gradients(zip(grads, self.student_model.trainable_weights))

            # Evaluate student model
            acc = self.evaluate(self.student_model, x_test, y_test)
            print(f"Student Model Accuracy: {acc:.4f}")

    @staticmethod
    def evaluate(model, x_test, y_test):
        results = model.evaluate(x_test, y_test, verbose=0)
        return results[1]

# Deep Mutual Learning
class DeepMutualLearning:
    def __init__(self, model_1, model_2, alpha=0.5):
        self.model_1 = model_1
        self.model_2 = model_2
        self.alpha = alpha

    def train(self, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        optimizer_1 = optimizers.Adam()
        optimizer_2 = optimizers.Adam()

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for step, (x_batch, y_batch) in enumerate(dataset):
                y_batch_onehot = tf.one_hot(tf.squeeze(y_batch), depth=10)

                with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
                    # Forward pass
                    logits_1 = self.model_1(x_batch, training=True)
                    logits_2 = self.model_2(x_batch, training=True)

                    # Supervised loss
                    loss_1 = losses.categorical_crossentropy(y_batch_onehot, logits_1)
                    loss_2 = losses.categorical_crossentropy(y_batch_onehot, logits_2)

                    # Mimicry loss
                    mimicry_loss_1 = kl_divergence_loss(tf.nn.softmax(logits_2), tf.nn.softmax(logits_1))
                    mimicry_loss_2 = kl_divergence_loss(tf.nn.softmax(logits_1), tf.nn.softmax(logits_2))

                    # Combined loss
                    total_loss_1 = self.alpha * loss_1 + (1 - self.alpha) * mimicry_loss_1
                    total_loss_2 = self.alpha * loss_2 + (1 - self.alpha) * mimicry_loss_2

                # Compute gradients
                grads_1 = tape_1.gradient(total_loss_1, self.model_1.trainable_weights)
                grads_2 = tape_2.gradient(total_loss_2, self.model_2.trainable_weights)

                # Apply gradients
                optimizer_1.apply_gradients(zip(grads_1, self.model_1.trainable_weights))
                optimizer_2.apply_gradients(zip(grads_2, self.model_2.trainable_weights))

            # Evaluate models
            acc_1 = self.evaluate(self.model_1, x_test, y_test)
            acc_2 = self.evaluate(self.model_2, x_test, y_test)
            print(f"Model 1 Accuracy: {acc_1:.4f}, Model 2 Accuracy: {acc_2:.4f}")

    @staticmethod
    def evaluate(model, x_test, y_test):
        results = model.evaluate(x_test, y_test, verbose=0)
        return results[1]

# Instantiate models
teacher_model = create_model()
student_model = create_model()
dml_model_1 = create_model()
dml_model_2 = create_model()
codis_model_1 = create_model()
codis_model_2 = create_model()

# Train models
print("\nApplying Knowledge Distillation:")
kd = KnowledgeDistillation(teacher_model, student_model, temperature=3.0, alpha=0.5)
kd.train(x_train, y_train, x_test, y_test)
kd_acc = kd.evaluate(student_model, x_test, y_test)

print("\nApplying Deep Mutual Learning:")
dml = DeepMutualLearning(dml_model_1, dml_model_2, alpha=0.5)
dml.train(x_train, y_train, x_test, y_test)
dml_acc_1 = dml.evaluate(dml_model_1, x_test, y_test)
dml_acc_2 = dml.evaluate(dml_model_2, x_test, y_test)

print("\nApplying Co-Distillation:")
codis = CoDistillation(codis_model_1, codis_model_2, temperature=3.0, alpha=0.5)
codis.train(x_train, y_train, x_test, y_test)
codis_acc_1 = codis.evaluate(codis_model_1, x_test, y_test)
codis_acc_2 = codis.evaluate(codis_model_2, x_test, y_test)

# Plot results
methods = ['Knowledge Distillation', 'Deep Mutual Learning', 'Co-Distillation']
accuracies = [kd_acc, (dml_acc_1 + dml_acc_2) / 2, (codis_acc_1 + codis_acc_2) / 2]

plt.figure(figsize=(10, 6))
plt.bar(methods, accuracies, color=['blue', 'green', 'orange'])
plt.title('Comparison of Distillation Methods on CIFAR-10')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=12)
plt.show()
