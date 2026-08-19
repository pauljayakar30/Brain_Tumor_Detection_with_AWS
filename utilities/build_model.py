from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import tensorflow as tf

def build_model():
    print(tf.config.list_physical_devices('GPU'))
    model = Sequential([
        Conv2D(32, kernel_size=(2,2), input_shape=(128, 128, 3), padding="Same"),
        Conv2D(32, kernel_size=(2,2), activation='relu', padding='Same'),
        
        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(2,2), activation='relu', padding='Same'),
        Conv2D(64, kernel_size=(2,2), activation='relu', padding='Same'),

        BatchNormalization(),
        MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        Dropout(0.25),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    model.summary()

    return model