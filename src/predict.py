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

from utils import *

pc = ProjectConfig()
MODEL_DIR = Path("./model_saves").resolve()
MODEL_NAME = "efficientnet_v2m"
MODEL_PATH = MODEL_DIR.joinpath(MODEL_NAME + ".h5")
model_path = pc.project_root_dir.joinpath(MODEL_PATH)
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")
print(f"BEGIN loading Model: {model_path}")
model = keras.models.load_model(model_path)
model.trainable = False
print(f"END loading Model {model_path}")
data_dir = pc.data_root_dir.joinpath("training_data")
image_to_predict_path = data_dir.joinpath(
    "mushroom/fa0f0384f8c20eadcf8bd25b69d12073a5d48c91271077bc58a722f2dc9198b4.png"
)
class_list = joblib.load("class_list.lzma")

print("BEGIN predicting")
img = keras.utils.load_img(image_to_predict_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
best_pred = np.argmax(preds)
predicted_class = class_list[best_pred]
print(f"Predicted: {predicted_class}: {preds[0][best_pred]}")
