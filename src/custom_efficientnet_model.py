import os
import pickle
import platform
from pathlib import Path

import joblib
import keras
import numpy as np
import tensorflow as tf
from keras.applications.efficientnet_v2 import (
    EfficientNetV2M,
    decode_predictions,
    preprocess_input,
)
from platformdirs import user_documents_dir
from tensorflow.keras import initializers
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ProjectConfig:
    def __init__(self):
        self.system = platform.system()
        self.user_documents_dir = Path(user_documents_dir())
        if platform.system() == "Windows":
            self.class_root_dir = self.user_documents_dir.joinpath("01-Berkeley/210")
            self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
            self.data_root_dir = self.class_root_dir.joinpath("data")
        elif platform.system() == "Linux":
            self.class_root_dir = Path("/tf/notebooks")
            self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
            self.data_root_dir = self.class_root_dir.joinpath("data")


def main():
    """
    Stage this for transfer learning since we have an extremely small dataset for now
    """
    pc = ProjectConfig()
    training_dir_path = pc.data_root_dir.joinpath("extracted_common_images")
    NUM_CLASSES = 0
    for i in training_dir_path.iterdir():
        if i.is_dir():
            NUM_CLASSES += 1
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        training_dir_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        subset="training",
    )
    validation_generator = train_datagen.flow_from_directory(
        training_dir_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        subset="validation",
    )
    class_list = list(train_generator.class_indices.keys())
    joblib.dump(class_list, "class_list.lzma", compress=3, protocol=pickle.HIGHEST_PROTOCOL)

    base_model = EfficientNetV2M(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    # Modify the output layer
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu", kernel_initializer=initializers.HeNormal())(x)
    # x = Dense(512, activation='relu', kernel_initializer=initializers.HeNormal())(x)
    # x = Dense(128, activation='relu', kernel_initializer=initializers.HeNormal())(x)
    # x = Dense(64, activation='relu', kernel_initializer=initializers.HeNormal())(x)
    predictions = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples
        // validation_generator.batch_size,
    )
    joblib.dump(
        history.history, "history.lzma", compress=3, protocol=pickle.HIGHEST_PROTOCOL
    )
    model.save("efficientnet_v2m.h5", save_format="h5")


if __name__ == "__main__":
    main()
