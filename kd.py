try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, losses, optimizers, datasets
    import numpy as np
except ModuleNotFoundError as e:
    raise ImportError("TensorFlow is not installed. Please ensure TensorFlow is installed in your environment.") from e

# Load MNIST data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Define the teacher model
def create_teacher_model():
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define the student model
def create_student_model():
    model = models.Sequential([
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Train the teacher model
teacher_model = create_teacher_model()
teacher_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
teacher_model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# Generate soft targets from the teacher model
def generate_soft_targets(model, data, temperature):
    logits = model.predict(data) / temperature
    soft_targets = tf.nn.softmax(logits).numpy()
    return soft_targets

# Define distillation loss
class DistillationLoss(losses.Loss):
    def __init__(self, temperature, alpha):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def call(self, y_true, y_pred):
        hard_loss = losses.sparse_categorical_crossentropy(y_true, y_pred)
        soft_loss = losses.categorical_crossentropy(y_true, y_pred)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Train the student model with knowledge distillation
student_model = create_student_model()
temperature = 5.0
alpha = 0.5
soft_targets = generate_soft_targets(teacher_model, x_train, temperature)

student_model.compile(
    optimizer='adam',
    loss=DistillationLoss(temperature, alpha),
    metrics=['accuracy']
)

student_model.fit(
    x_train,
    soft_targets,
    epochs=5,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# Evaluate the student model
student_model.evaluate(x_test, y_test)
