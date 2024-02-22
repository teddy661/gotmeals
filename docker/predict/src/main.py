from datetime import datetime
from io import BytesIO
from pathlib import Path

import joblib
import keras
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from keras.applications.efficientnet_v2 import (
    EfficientNetV2M,
    decode_predictions,
    preprocess_input,
)
from PIL import Image
from pydantic import BaseModel, ConfigDict

missing_app_version = False
try:
    from src.app_version import get_app_version
except ImportError:
    print("app_version file has not been generated. /version endpoint will be broken")
    missing_app_version = True

script_path = Path(__file__).parent.absolute()
sk_model_file = script_path.joinpath("efficientnet_v2m.h5")
class_list = script_path.joinpath("class_list.lzma")
if sk_model_file.exists():
    if sk_model_file.is_file():
        model = tf.keras.models.load_model(sk_model_file)
        model.trainable = False
    else:
        print(f" {sk_model_file} isn't a file!")
else:
    print(f" {sk_model_file} doesn't exist!")
if class_list.exists():
    if class_list.is_file():
        class_list = joblib.load(class_list)
    else:
        print(f" {class_list} isn't a file!")


app = FastAPI()


class PredictResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Ingredient: str
    Confidence: float


@app.get("/")
async def root():
    """Return 501 Not Implemented for the root endpoint. Do nothing else."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="not implemented"
    )


@app.get("/health", status_code=status.HTTP_200_OK)
async def return_health():
    """Return 200. Do nothing else."""
    return {"health": "ok"}


@app.get("/hello", status_code=status.HTTP_200_OK)
async def say_hello(name: str):
    """Return 422 Bad Request if name is not specified as a query parameter.
    Otherwise, return 200 and a json message of "hello [value].
    """
    return {"message": f"hello {name}"}


@app.get("/version", status_code=status.HTTP_200_OK)
async def return_git_version():
    if missing_app_version:
        return {"git-version": "unknown"}
    else:
        return get_app_version()


def read_imagefile(file) -> Image.Image:
    image = keras.utils.load_img(BytesIO(file), target_size=(224, 224))
    return image


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(file: UploadFile = File(...)):
    """Return 422 Bad Request if data is not specified as a query parameter.
    Otherwise, return 200 and a json message of "hello [value].
    """
    image = read_imagefile(await file.read())
    image = keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    prediction = model.predict(image)
    best_pred_index = np.argmax(prediction)
    predicted_class = class_list[best_pred_index]
    final_result = PredictResult(
        Ingredient=predicted_class, Confidence=prediction[0][best_pred_index]
    )
    return final_result
