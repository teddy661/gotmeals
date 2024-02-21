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

    pc = ProjectConfig()
    common_dataset_images_path = pc.data_root_dir.joinpath(
        "common_ingredient_images_dataset.parquet"
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    if common_dataset_images_path.exists() and not args.force:
        logging.error(f"File {common_dataset_images_path} already exists.")
        exit(1)
    elif common_dataset_images_path.exists() and args.force:
        common_dataset_images_path.unlink(missing_ok=True)

    common_columns = [
        ("ClassId", pl.Utf8),
        ("ImageId", pl.Utf8),
        ("Image_Path", pl.Utf8),
        ("Width", pl.Int64),
        ("Height", pl.Int64),
        ("Resolution", pl.Int64),
    ]
    common_df = pl.DataFrame({}, schema=common_columns)
    common_ingredients = pl.read_parquet("common_ingredients.parquet")
    parquet_files = list(pc.data_root_dir.glob("*.parquet"))
    parquet_files.remove(pc.data_root_dir.joinpath("RecipeNLG_dataset.parquet"))

    for parquet_file in parquet_files:
        logging.info(f"Processing Parquet:\t{parquet_file}")
        df = pl.read_parquet(parquet_file)
        for row in common_ingredients.rows(named=True):
            df_filtered = df.filter(pl.col("ClassId") == row["ClassId"])
            if df_filtered.height > 0:
                common_df = pl.concat([common_df, df_filtered])
            logging.info(
                f"Ingredient:\t{row['ClassId']:15}\tAdded:\t{df_filtered.height:-4d} rows"
            )
        del df
        del df_filtered

    common_df.write_parquet(
        common_dataset_images_path,
        compression="uncompressed",
    )


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
