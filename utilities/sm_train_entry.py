import argparse
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
)
from tensorflow.keras.models import Sequential


def load_images_from_folder(folder_path, label, encoder, data, result):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not (file.endswith(".jpg") or file.endswith(".jpeg")):
                continue
            img_path = os.path.join(root, file)
            img = Image.open(img_path).convert("RGB").resize((128, 128))
            img = np.array(img)
            if img.shape != (128, 128, 3):
                continue
            data.append(img)
            result.append(encoder.transform([[label]]).toarray())


def prepare(data_dir):
    data, result = [], []
    encoder = OneHotEncoder()
    encoder.fit([[0], [1]])
    load_images_from_folder(os.path.join(data_dir, "yes"), 0, encoder, data, result)
    load_images_from_folder(os.path.join(data_dir, "no"),  1, encoder, data, result)
    return np.array(data), np.vstack(result)


def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding="Same"),
        Conv2D(32, kernel_size=(2, 2), activation="relu", padding="Same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(2, 2), activation="relu", padding="Same"),
        Conv2D(64, kernel_size=(2, 2), activation="relu", padding="Same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="Adamax", metrics=["accuracy"])
    model.summary()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=42)
    parser.add_argument("--model-dir",  type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training",   type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "./dataset"))
    args = parser.parse_args()

    print(f"Data dir  : {args.training}")
    print(f"Model dir : {args.model_dir}")
    print(f"Epochs    : {args.epochs}  |  Batch size: {args.batch_size}")
    print("GPUs available:", tf.config.list_physical_devices("GPU"))

    data, result = prepare(args.training)
    x_train, x_test, y_train, y_test = train_test_split(
        data, result, test_size=0.2, shuffle=True, random_state=42
    )

    model = build_model()
    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    save_path = os.path.join(args.model_dir, "1")
    model.save(save_path, save_format="tf")
    print(f"Model saved to {save_path}")
