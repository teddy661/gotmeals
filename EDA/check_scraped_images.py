import logging
import multiprocessing as mp
import re
import sys
from pathlib import Path

import polars as pl
from blake3 import blake3

from utils import *

pc = ProjectConfig()

TARGET_DIR_1 = pc.data_root_dir.joinpath(
    "image_downloads_20240302-132212/image_downloads_20240302-132212"
)
TARGET_PARQUET_FILE_1 = TARGET_DIR_1.parent.joinpath(
    "image_downloads_20240302-132212.parquet"
)

TARGET_DIR_2 = pc.data_root_dir.joinpath(
    "image_downloads_20240303-091315/image_downloads_20240303-091315"
)
TARGET_PARQUET_FILE_2 = TARGET_DIR_2.parent.joinpath(
    "image_downloads_20240303-091315.parquet"
)


def recurse_dir(target_dir: Path):
    all_images = []
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist.")
    else:
        for item in target_dir.iterdir():
            if item.is_file():
                if item.suffix in [".jpg", ".png", ".jpeg", ".gif"]:
                    all_images.append(str(item.name))
            if item.is_dir():
                all_images.extend(recurse_dir(item))
    return all_images


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    logging.info("BEGIN: Reading parquet files")
    image_df_1 = pl.read_parquet(TARGET_PARQUET_FILE_1)
    image_df_2 = pl.read_parquet(TARGET_PARQUET_FILE_2)
    all_images_df = pl.concat([image_df_1, image_df_2])
    logging.info("END  : Reading parquet files")

    if not TARGET_DIR_1.exists() or not TARGET_DIR_2.exists():
        print(f"Directory {TARGET_DIR_1} or {TARGET_DIR_2} does not exist.")
        exit(1)
    if not TARGET_PARQUET_FILE_1.exists() or not TARGET_PARQUET_FILE_2.exists():
        print(
            f"File {TARGET_PARQUET_FILE_1} or {TARGET_PARQUET_FILE_2} does not exist."
        )
        exit(1)
    logging.info("BEGIN: Enumerating images in target directories")
    for target_dir in [TARGET_DIR_1, TARGET_DIR_2]:
        all_images = recurse_dir(target_dir)
    logging.info("END  : Enumerating images in target directories")

    print(f"Total images: {len(all_images)}")
    for image in all_images:
        num_rows = all_images_df.filter(pl.col("ImageId") == image).height
        if num_rows == 0:
            logging.error(f"NOT FOUND: {image}")
        elif num_rows > 1:
            logging.warning(f"DUPLICATED: {num_rows} {image}")
        elif num_rows == 1:
            pass


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
