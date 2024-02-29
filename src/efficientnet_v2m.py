from pathlib import Path

import keras
import numpy as np
from keras.applications.efficientnet_v2 import (
    EfficientNetV2M,
    decode_predictions,
    preprocess_input,
)

from utils import *

model = EfficientNetV2M(weights="imagenet")
# img_path = Path(
#    "../../data/FoodClassification/train_images/train_images/ff9aad660b.jpg"
# )
img_path = Path(
    "../../data/FoodClassification/train_images/train_images/00193b659c.jpg"
)
img = keras.utils.load_img(img_path, target_size=(480, 480))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
print("Predicted:", decode_predictions(preds, top=3)[0])
