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

Image.MAX_IMAGE_PIXELS = 110000000
CLASSIFICATION_ROOT = pc.project_root_dir.joinpath("EDA/Image_Scraping")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(description="Parse Unsplash dataset")
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
    TRAIN_IMG_DIR = CLASSIFICATION_ROOT.joinpath("Scraped_images")

    target_parquet_file = pc.data_root_dir.joinpath("UnsplashDataset.parquet")
    if target_parquet_file.exists() and not args.force:
        logging.error(f"File {target_parquet_file} already exists.")
        exit(1)
    elif target_parquet_file.exists() and args.force:
        target_parquet_file.unlink(missing_ok=True)

    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8

    train_df = pl.read_csv(
        CLASSIFICATION_ROOT.joinpath("Unsplash_Attributes_2.csv"), has_header=True
    )
    train_df = train_df.rename({"Image_Labels_jpg": "ImageId", "Labels": "ClassId"})
    train_df = train_df.drop(["Image_Labels"])
    train_df = train_df.with_columns(
        pl.col("ImageId")
        .map_elements(lambda x: update_path(x, TRAIN_IMG_DIR))
        .alias("Image_Path")
    )
    # df = df.filter(pl.col("In_Top_100") == 1)  # Only include top 100 ingredients This column was removed from the csv
    df = parallelize_dataframe(train_df, read_image_wrapper, num_cpus)
    df = df.select(
        pl.col("ClassId"),
        pl.col("ImageId"),
        pl.col("Image_Path"),
        pl.col("Width"),
        pl.col("Height"),
        pl.col("Resolution"),
    )

    logging.info("Remove rows with missing images")
    mask = df["Image_Path"].map_elements(lambda f: Path(f).exists())
    final_filtered_df = df.filter(mask)
    removed_rows = df.filter(~mask)
    print(removed_rows)
    logging.info("Writing parquet file")
    final_filtered_df.write_parquet(
        target_parquet_file, compression="lz4", compression_level=3
    )


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
