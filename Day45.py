import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential(
    [
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into 1D
        Dense(128, activation="relu"),  # Hidden layer
        Dense(64, activation="relu"),  # Hidden layer
        Dense(10, activation="softmax"),  # Output layer
    ]
)

model.compile(
    optimizer="adam",  # Using Adam optimizer
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1
)

print("history=", history.history["accuracy"])
print("history=", history.history["val_accuracy"])

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# predict on test data
y_pred = model.predict(X_test)
# Visualize predictions
for index, (image, label) in enumerate(zip(X_test[:8], y_pred[:8])):
    plt.subplot(2, 4, index + 1)
    plt.imshow(image.reshape(28, 28), cmap="gray")
    print(label)
    # plt.title(f"Predicted=, {label}")
plt.show()
