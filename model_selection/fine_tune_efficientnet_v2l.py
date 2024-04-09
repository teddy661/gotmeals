import os
import pickle
import platform
from pathlib import Path

import joblib
import keras
import numpy as np
import tensorflow as tf
from keras.applications.efficientnet_v2 import (
    EfficientNetV2L,
    preprocess_input,
)
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
    training_dir_path = Path("/data/fc/train")
    validation_dir_path = Path("/data/fc/valid")
    NUM_CLASSES = 0
    NUM_EPOCHS = 1000
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00001  # Default is 0.001; 1e-5 for fine-tuning
    MODEL_DIR = Path("./model_fine_tuned_saves_fc").resolve()
    MODEL_NAME = "EfficientNetV2L"
    TIME_FILE_NAME = MODEL_NAME + "_TRAINING_TIME"

    gpus = tf.config.list_physical_devices("GPU")
    text_gpu_list = [x.name.replace("/physical_device:", "") for x in gpus]

    mirrored_strategy = tf.distribute.MirroredStrategy(devices=text_gpu_list)

    if len(gpus) > 0:
        BATCH_SIZE = BATCH_SIZE * len(gpus)
    else:
        print("No GPUs detected. Exiting")
        exit(1)

    validation_split = 0.2
    flow_from_directory_seed = 42
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # rescale=1.0 / 255, # DO NOT TURN THIS ON FOR EFFICIENT NET! BAD THINGS
        rotation_range=40,  # Random rotations from 0 to 40 degrees
        width_shift_range=0.2,  # Random horizontal shifts (as a fraction of total width)
        height_shift_range=0.2,  # Random vertical shifts (as a fraction of total height)
        shear_range=0.2,  # Shear transformation intensity
        zoom_range=0.2,  # Random zoom range
        horizontal_flip=True,  # Randomly flip inputs horizontally
        vertical_flip=True,  # Randomly flip inputs vertically
        fill_mode="wrap",  # Strategy for filling in newly created pixels
        validation_split=validation_split,
    )

    # Create a new ImageDataGenerator instance for validation data without augmentation
    # but with the necessary preprocessing.
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        # No augmentation, only preprocessing
    )

    train_generator = train_datagen.flow_from_directory(
        training_dir_path,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=flow_from_directory_seed,
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir_path,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=True,
        seed=flow_from_directory_seed,
    )

    with mirrored_strategy.scope():
        print(f"Loading model {MODEL_NAME}...")
        model = tf.keras.models.load_model(f"model_saves_fc/{MODEL_NAME}.h5")
        for layer in model.layers:
            layer.trainable = True
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

        model.summary(show_trainable=True)
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        verbose=1,
        patience=5,
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
    end_time = datetime.now()
    total_training_time = end_time - start_time
    joblib.dump(
        total_training_time,
        f"{MODEL_DIR}/{TIME_FILE_NAME}.lzma",
        compress=3,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    print(f"Training time for {MODEL_NAME}: {total_training_time}")


if __name__ == "__main__":
    main()
