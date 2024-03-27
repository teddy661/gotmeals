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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from utils import *


def main():
    """
    Stage this for transfer learning since we have an extremely small dataset for now
    """
    pc = ProjectConfig()
    training_dir_path = pc.data_root_dir.joinpath("sampled_training_data")
    NUM_CLASSES = 0
    NUM_EPOCHS = 1000
    BATCH_SIZE = 20
    LEARNING_RATE = 0.00001  # Default is 0.001; 1e-5 for fine-tuning
    MODEL_DIR = Path("./model_fine_tuned_saves").resolve()
    MODEL_NAME = "efficientnet_v2m"

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # rescale=1.0 / 255, # DO NOT TURN THIS ON FOR EFFICIENT NET! BAD THINGS
        rotation_range=40,  # Random rotations from 0 to 40 degrees
        width_shift_range=0.2,  # Random horizontal shifts (as a fraction of total width)
        height_shift_range=0.2,  # Random vertical shifts (as a fraction of total height)
        shear_range=0.2,  # Shear transformation intensity
        zoom_range=0.2,  # Random zoom range
        horizontal_flip=True,  # Enable random horizontal flips
        fill_mode="nearest",  # Strategy for filling in newly created pixels
        validation_split=0.2,
    )

    train_generator = train_datagen.flow_from_directory(
        training_dir_path,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="training",
        shuffle=True,
        seed=42,
    )

    validation_generator = train_datagen.flow_from_directory(
        training_dir_path,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        subset="validation",
        shuffle=True,
        seed=42,
    )

    model = tf.keras.models.load_model("model_saves/efficientnet_v2m.h5")
    for layer in model.layers:
        layer.trainable = True
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    model.summary(show_trainable=True)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss=SparseCategoricalCrossentropy(), metrics=["accuracy"]
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=1,
        patience=10,
        min_delta=0.01,
        restore_best_weights=True,
    )
    model_checkpoint = ModelCheckpoint(
        "model_checkpoints/{epoch:04d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        verbose=1,
        save_weights_only=False,
        save_best_only=False,
    )
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples
        // validation_generator.batch_size,
        verbose=1,
        callbacks=[early_stopping],
    )
    if MODEL_DIR.exists() is False:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR.joinpath(MODEL_NAME + "_history"), "wb") as history_file:
        pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)
    model.save(
        MODEL_DIR.joinpath(MODEL_NAME + ".h5"),
        save_format="h5",
        overwrite=True,
        include_optimizer=True,
    )


if __name__ == "__main__":
    main()
