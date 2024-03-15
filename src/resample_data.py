import argparse
import logging
import multiprocessing as mp
import platform
import shutil
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from platformdirs import user_documents_dir

from utils import *

SAMPLES_PER_CLASS = 120
RANDOM_SEED = 42
TRAIN_PERCENTAGE = 0.8


def get_sampled_data(raw_train_df: pl.DataFrame) -> pl.DataFrame:
    """
    Oversample our data to a median class size of SAMPLES_PER_CLASS samples if the class has more than SAMPLES_PER_CLASS
    samples, then sample without replacement if it has less, sample with replacement.
    :param df: The dataframe to sample
    :return: The sampled dataframe
    """
    pl.set_random_seed(RANDOM_SEED)
    train_equal_sample_df = pl.concat(
        [
            (
                x.sample(SAMPLES_PER_CLASS, with_replacement=True, shuffle=False)
                if x.height <= SAMPLES_PER_CLASS
                else x.sample(SAMPLES_PER_CLASS, with_replacement=False, shuffle=False)
            )
            for x in raw_train_df.partition_by("ClassId")
        ]
    )
    return train_equal_sample_df


def duplicate_image(target_dir: str, class_id: str, image_path: str) -> pl.DataFrame:
    """ """
    target_dir = Path(target_dir)
    src_image_path = Path(image_path)
    src_image_base_name = src_image_path.stem
    src_image_suffix = src_image_path.suffix
    random_append = uuid.uuid4().hex
    image_dir = target_dir.joinpath(class_id)
    image_dir.mkdir(parents=True, exist_ok=True)
    target_image_path = image_dir.joinpath(
        src_image_base_name + "_" + random_append + src_image_suffix
    )
    if src_image_path.exists() and not src_image_path.is_file():
        logging.error(f"File {src_image_path} is not a file")
        return None
    try:
        shutil.copy(src_image_path, target_image_path)
    except IOError as e:
        logging.error(f"Error copying {src_image_path} to {target_image_path}: {e}")
        return None
    # print(f"Copying {src_image_path} to {target_image_path}")
    return str(target_image_path)


def create_sampled_data(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.struct(["target_dir", "ClassId", "Scaled_Image_Path"])
        .map_elements(
            lambda x: duplicate_image(
                x["target_dir"], x["ClassId"], x["Scaled_Image_Path"]
            ),
        )
        .alias("Sampled_Image_Path")
    )
    return df


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Parse merged dataset into training data."
    )
    parser.add_argument(
        "-f",
        dest="force",
        help="force remove existing training data directory and recreate it",
        action="store_true",
    )
    args = parser.parse_args()
    prog_name = parser.prog
    # Blindly oversample the data to ensure consistent class distribution
    # we can be smarter later on
    # Load the data
    pc = ProjectConfig()
    target_name = "sampled_training_data"
    TARGET_DIR = pc.data_root_dir.joinpath(target_name)
    TARGET_PARQUET = pc.data_root_dir.joinpath(target_name + ".parquet")
    SOURCE_DATA = pc.data_root_dir.joinpath("training_data.parquet")

    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Source data {SOURCE_DATA} does not exist")
        exit(1)

    # Read data and drop any images that we're not scaled
    source_df = pl.read_parquet(SOURCE_DATA)
    original_count = source_df.height
    filtered_source_df = source_df.filter(pl.col("Scaled_Image_Path") != "")
    filtered_count = filtered_source_df.height
    logging.warning(
        f"Filtered {original_count - filtered_count} rows with no scaled image"
    )

    sampled_data = get_sampled_data(filtered_source_df)
    sampled_data = sampled_data.with_columns(
        pl.lit(str(TARGET_DIR)).alias("target_dir")
    )

    if not TARGET_DIR.exists():
        TARGET_DIR.mkdir(parents=True)
        if TARGET_PARQUET.exists():
            TARGET_PARQUET.unlink(missing_ok=True)
    elif args.force:
        logging.info(f"Removing {TARGET_DIR} and parquet file")
        shutil.rmtree(TARGET_DIR)
        TARGET_PARQUET.unlink(missing_ok=True)
    elif not args.force:
        logging.error(f"Directory {TARGET_DIR} already exists")
        exit(1)

    TARGET_DIR.mkdir(parents=True)
    if not TARGET_DIR.exists():
        logging.error(f"Directory {TARGET_DIR} does not exist and it should")
        exit(2)

    sampled_data = parallelize_dataframe(sampled_data, create_sampled_data, 8)
    sampled_data = sampled_data.drop("target_dir")

    sampled_data.write_parquet(TARGET_PARQUET)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
