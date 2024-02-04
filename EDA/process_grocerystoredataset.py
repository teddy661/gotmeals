import argparse
import gc
import logging
import multiprocessing as mp
import sys
from datetime import datetime
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import psutil
from PIL import Image

from utils import convert_numpy_to_bytesio, parallelize_dataframe

DATA_ROOT = Path("C:/Users/teddy/Documents/01-Berkeley/210/data")
CLASSIFICATION_ROOT = DATA_ROOT.joinpath(
    "GroceryStoreDataset/GroceryStoreDataset/dataset"
)


def update_path(path: Path, root_dir: Path) -> str:
    return str(root_dir.joinpath(path).resolve())


def read_image(image_path: Path) -> tuple:
    image_data = (
        np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    )
    image_height = image_data.shape[0]
    image_width = image_data.shape[1]
    image_resolution = image_height * image_width
    return (
        image_width,
        image_height,
        image_resolution,
        convert_numpy_to_bytesio(image_data),
    )


def read_image_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one reads an image from disk and stores it as a flattened list in a column.
    We convert the image to float32 and normalize it to the range of 0-1. cvtColor is which
    we use extensively expects thing in uint8 format. We'll convert back to float32.
    The meta images are png with 4 channels add .convert('RGB') to convert to 3 channels
    doesn't affect the existing jpg
    """
    df = df.with_columns(
        pl.col("Image_Path")
        .map_elements(
            lambda x: dict(
                zip(
                    ("Width", "Height", "Resolution", "Image_Data"),
                    read_image(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    return df


def main():
    if not CLASSIFICATION_ROOT.exists():
        logging.error(f"Directory {CLASSIFICATION_ROOT} does not exist.")
    TRAIN_IMG_DIR = CLASSIFICATION_ROOT.joinpath("dataset/train")

    num_cpus = 8

    train_df = pl.read_csv(CLASSIFICATION_ROOT.joinpath("train.txt"), has_header=False)
    train_df = train_df.rename(
        {
            "column_1": "Image_Path",
            "column_2": "Class ID (int)",
            "column_3": "Coarse Class ID (int)",
        }
    )
    classes_df = pl.read_csv(CLASSIFICATION_ROOT.joinpath("classes.csv"))
    df = train_df.join(classes_df, on="Class ID (int)")
    df = df.drop("Coarse Class ID (int)_right")
    df = df.with_columns(
        pl.col("Image_Path")
        .map_elements(lambda x: update_path(x, CLASSIFICATION_ROOT))
        .alias("Image_Path")
    )
    df = parallelize_dataframe(df, read_image_wrapper, num_cpus)
    print(df.head())
    df.write_parquet("grocerystoredataset.parquet", compression="snappy")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
