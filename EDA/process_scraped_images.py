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


CLASSIFICATION_ROOT_1 = pc.data_root_dir.joinpath("image_downloads_20240302-132212")
CLASSIFICATION_ROOT_2 = pc.data_root_dir.joinpath("image_downloads_20240303-091315")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Parse Scraped dataset")
    parser.add_argument(
        "-f",
        dest="force",
        help="force overwrite of existing parquet files",
        action="store_true",
    )
    args = parser.parse_args()
    prog_name = parser.prog

    target_parquet_file = pc.data_root_dir.joinpath("scraped_images.parquet")
    if target_parquet_file.exists() and not args.force:
        logging.error(f"File {target_parquet_file} already exists.")
        exit(1)
    elif target_parquet_file.exists() and args.force:
        target_parquet_file.unlink(missing_ok=True)

    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8

    data_frame_list = []
    for class_root in [CLASSIFICATION_ROOT_1, CLASSIFICATION_ROOT_2]:
        if not class_root.exists():
            logging.error(f"Directory {class_root} does not exist.")
            continue
        logging.info(f"Processing directory: {class_root.name}")
        TRAIN_IMG_DIR = class_root.joinpath(class_root.name)
        ClassId = []
        ImageId = []
        ImagPath = []
        directories = [f for f in TRAIN_IMG_DIR.iterdir() if f.is_dir()]
        for d in directories:
            files = [
                f
                for f in d.iterdir()
                if f.is_file() and f.suffix in [".jpg", ".png", ".jpeg"]
            ]
            for f in files:
                ClassId.append(d.name)
                ImageId.append(f.name)
                ImagPath.append(str(f))
        df = pl.DataFrame(
            {"ClassId": ClassId, "ImageId": ImageId, "Image_Path": ImagPath}
        )
        df = parallelize_dataframe(df, read_image_wrapper, num_cpus)
        logging.info(f"Captured {df.height} images")
        data_frame_list.append(df)

    final_df = pl.concat(data_frame_list)

    print(final_df.head())
    final_df.write_parquet(target_parquet_file, compression="snappy")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
