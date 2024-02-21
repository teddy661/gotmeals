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

from utils import *

pc = ProjectConfig()

CLASSIFICATION_ROOT = pc.data_root_dir.joinpath("FoodClassification")


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
    )
    print(df.head())
    df.write_parquet(target_parquet_file, compression="lz4", compression_level=3)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
