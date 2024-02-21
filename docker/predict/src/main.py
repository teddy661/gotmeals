from datetime import datetime
from pathlib import Path

import joblib
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

missing_app_version = False
try:
    from src.app_version import get_app_version
except ImportError:
    print("app_version file has not been generated. /version endpoint will be broken")
    missing_app_version = True

script_path = Path(__file__).parent.absolute()
sk_model_file = script_path.joinpath("efficientnet_v2m.h5")
if sk_model_file.exists():
    if sk_model_file.is_file():
        model = tf.keras.models.load_model(sk_model_file)
        model.trainable = False
    else:
        print(f" {sk_model_file} isn't a file!")
else:
    print(f" {sk_model_file} doesn't exist!")

app = FastAPI()


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
