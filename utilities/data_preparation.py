import os
from PIL import Image
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_images_from_folder(folder_path, label, encoder, data, result):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not (file.endswith(".jpg") or file.endswith(".jpeg")):
                continue

            img_path = os.path.join(root, file)
            img = Image.open(img_path).resize((128, 128))
            img = np.array(img)

            if img.shape != (128, 128, 3):
                continue

            data.append(img)
            result.append(encoder.transform([[label]]).toarray())


def prepare():
    data = []
    result = []

    encoder = OneHotEncoder()
    encoder.fit([[0], [1]])

    load_images_from_folder("./dataset/yes", 0, encoder, data, result)
    load_images_from_folder("./dataset/no", 1, encoder, data, result)

    return np.array(data), np.vstack(result)


def split_data(data, result):
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=42)

    return (x_train, 
            x_test, 
            y_train, 
            y_test
            )

