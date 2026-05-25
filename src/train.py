import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", category=UserWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
EPOCHS        = 15
BATCH_SIZE    = 256
LEARNING_RATE = 0.001
MODEL_PATH    = "model/mnist_model.keras"


def load_data():
    """Load and preprocess the MNIST dataset."""
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Split off a validation set and normalise pixel values to [0, 1]
    X_valid = X_train_full[:5000] / 255.0
    X_train = X_train_full[5000:] / 255.0
    y_valid = y_train_full[:5000]
    y_train = y_train_full[5000:]
    X_test  = X_test / 255.0

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def build_model():
    """Construct and compile the sequential model."""
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10,  activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

    model.summary()
    return model


def save_model(model):
    """Persist the trained model to disk."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def main():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()

    model   = build_model()
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_valid, y_valid),
    )

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=5000)
    print(f"\nTest results — Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    save_model(model)


if __name__ == "__main__":
    main()
