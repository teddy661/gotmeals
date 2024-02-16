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

from utils import ProjectConfig, convert_numpy_to_bytesio, parallelize_dataframe

pc = ProjectConfig()

CLASSIFICATION_ROOT = pc.data_root_dir.joinpath("FoodClassification")


def update_path(path: Path, root_dir: Path) -> str:
    return str(root_dir.joinpath(path).resolve())


def read_image(image_path: Path) -> tuple:
    image = Image.open(image_path)
    if image.format == "PNG":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    image_data = np.asarray(image, dtype=np.float32) / 255.0
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
    parser = argparse.ArgumentParser(description="Parse foodclassification dataset")
    parser.add_argument(
        "-f",
        dest="force",
        help="force overwrite of existing parquet files",
        action="store_true",
    )
    args = parser.parse_args()
    prog_name = parser.prog
    if not CLASSIFICATION_ROOT.exists():
        logging.error(f"Directory {CLASSIFICATION_ROOT} does not exist.")
        exit(1)
    TRAIN_IMG_DIR = CLASSIFICATION_ROOT.joinpath("train_images/train_images")

    target_parquet_file = pc.data_root_dir.joinpath("FoodClassification.parquet")
    if target_parquet_file.exists() and not args.force:
        logging.error(f"File {target_parquet_file} already exists.")
        exit(1)
    elif target_parquet_file.exists() and args.force:
        target_parquet_file.unlink(missing_ok=True)

    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8

    df = pl.read_csv(CLASSIFICATION_ROOT.joinpath("train_img.csv"))
    df = df.with_columns(
        pl.col("ImageId")
        .map_elements(lambda x: update_path(x, TRAIN_IMG_DIR))
        .alias("Image_Path")
    )
    df = parallelize_dataframe(df, read_image_wrapper, num_cpus)
    df = df.rename({"ClassName": "ClassId"})
    df = df.select(
        pl.col("ClassId"),
        pl.col("ImageId"),
        pl.col("Image_Path"),
        pl.col("Width"),
        pl.col("Height"),
        pl.col("Resolution"),
        pl.col("Image_Data"),
    )
    print(df.head())
    df.write_parquet(target_parquet_file, compression="lz4", compression_level=3)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
