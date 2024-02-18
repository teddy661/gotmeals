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


CLASSIFICATION_ROOT = pc.data_root_dir.joinpath(
    "FruitsClassification/Fruits Classification"
)


def main():
    parser = argparse.ArgumentParser(description="Parse FruitsClassification dataset")
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
    TRAIN_IMG_DIR = CLASSIFICATION_ROOT.joinpath("train")

    target_parquet_file = pc.data_root_dir.joinpath("FruitsClassification.parquet")
    if target_parquet_file.exists() and not args.force:
        logging.error(f"File {target_parquet_file} already exists.")
        exit(1)
    elif target_parquet_file.exists() and args.force:
        target_parquet_file.unlink(missing_ok=True)

    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8

    ClassId = []
    ImageId = []
    ImagPath = []
    directories = [f for f in TRAIN_IMG_DIR.iterdir() if f.is_dir()]
    for d in directories:
        files = [f for f in d.iterdir() if f.is_file()]
        for f in files:
            ClassId.append(d.name)
            ImageId.append(f.name)
            ImagPath.append(str(f))
    df = pl.DataFrame({"ClassId": ClassId, "ImageId": ImageId, "Image_Path": ImagPath})
    df = parallelize_dataframe(df, read_image_wrapper, num_cpus)
    print(df.head())
    df.write_parquet(target_parquet_file, compression="snappy")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
