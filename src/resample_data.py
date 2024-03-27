import argparse
import logging
import multiprocessing as mp
import platform
import shutil
import uuid
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from utils import *

IMAGE_COUNT_CUTOFF = 27
TEST_PERCENTAGE = 0.2


def sample_group(group: pl.DataFrame, fraction: float) -> pl.DataFrame:
    RANDOM_SEED = 42
    sample_size = max(
        1, int(len(group) * fraction)
    )  # Calculate sample size, ensure at least 1
    return group.sample(
        n=sample_size, with_replacement=False, shuffle=True, seed=RANDOM_SEED
    )


def get_equally_sampled_data(
    input_df: pl.DataFrame, samples_per_class: int
) -> pl.DataFrame:
    """
    Oversample our data to a median class size of samples_per_class samples if the class
    has more than SAMPLES_PER_CLASS samples, then sample without replacement if it has less,
    sample with replacement.
    :param df: The dataframe to sample
    :return: The sampled dataframe
    """
    RANDOM_SEED = 42
    pl.set_random_seed(RANDOM_SEED)
    equal_sample_df = pl.concat(
        [
            (
                x.sample(
                    samples_per_class,
                    with_replacement=True,
                    shuffle=True,
                    seed=RANDOM_SEED,
                )
                if x.height <= samples_per_class
                else x.sample(
                    samples_per_class,
                    with_replacement=False,
                    shuffle=True,
                    seed=RANDOM_SEED,
                )
            )
            for x in input_df.partition_by("ClassId")
        ]
    )
    return equal_sample_df


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
    """ """
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
    """ """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(
        description="Parse merged dataset into training and test data. We're using tensorflow to create the validation data from the training data"
    )
    parser.add_argument(
        "-f",
        dest="force",
        help="force remove existing test and training data directory and recreate it",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        dest="test_images",
        help="number of images to reserve for the test data",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-n",
        dest="samples_per_class",
        help="number of images sampled per class",
        type=int,
        default=120,
    )

    args = parser.parse_args()
    prog_name = parser.prog
    samples_per_class = args.samples_per_class

    # Cap the number of CPUs to 8 or the number of cpu cores on the machine
    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8

    pc = ProjectConfig()
    cfg = pl.Config()
    cfg.set_tbl_rows(2000)
    cfg.set_tbl_width_chars(200)
    cfg.set_fmt_str_lengths(200)

    # Blindly oversample the data to ensure consistent class distribution
    # we can be smarter later on
    # Load the data
    SOURCE_DATA = pc.data_root_dir.joinpath("training_data.parquet")
    if not SOURCE_DATA.exists():
        raise FileNotFoundError(f"Source data {SOURCE_DATA} does not exist")
        exit(1)

    sampled_test_data_name = "sampled_test_data"
    SAMPLED_TEST_DATA_DIR = pc.data_root_dir.joinpath(sampled_test_data_name)
    SAMPLED_TEST_DATA_PARQUET = pc.data_root_dir.joinpath(
        sampled_test_data_name + ".parquet"
    )

    sampled_training_data_name = "sampled_training_data"
    SAMPLED_TRAINING_DATA_DIR = pc.data_root_dir.joinpath(sampled_training_data_name)
    SAMPLED_TRAINING_DATA_PARQUET = pc.data_root_dir.joinpath(
        sampled_training_data_name + ".parquet"
    )

    # Drop any duplicate Images, We have a few
    source_df = pl.read_parquet(SOURCE_DATA)
    source_df = source_df.drop(["ImageId", "Image_Path"])
    pre_dedup_count = source_df.height
    source_df = source_df.filter(~source_df.is_duplicated())
    post_dedup_count = source_df.height
    if pre_dedup_count != post_dedup_count:
        logging.warning(f"Filtered {pre_dedup_count - post_dedup_count} duplicate rows")
    else:
        logging.info(f"No duplicate rows were filtered from the source data")

    # Read data and drop any images that we're not scaled
    original_count = source_df.height
    filtered_source_df = source_df.filter(pl.col("Scaled_Image_Path") != "")
    filtered_count = filtered_source_df.height
    if original_count != filtered_count:
        logging.warning(
            f"Filtered {original_count - filtered_count} rows with no scaled image"
        )
    else:
        logging.info(f"No rows were found with missing scaled images")

    counts = (
        filtered_source_df.group_by("ClassId")
        .agg(pl.count("ClassId").alias("count"))
        .sort("count", descending=True)
    )

    # Filter out classes with less than IMAGE_COUNT_CUTOFF images
    included_classes = counts.filter(pl.col("count") >= IMAGE_COUNT_CUTOFF)["ClassId"]
    mask = filtered_source_df["ClassId"].is_in(included_classes)
    filtered_source_df = filtered_source_df.filter(mask)

    # Get the Test Data
    sample_group_with_fraction = partial(sample_group, fraction=TEST_PERCENTAGE)
    test_df = filtered_source_df.group_by("ClassId").map_groups(
        sample_group_with_fraction
    )

    # Remove the Test Data From the Candidate Data
    remaining_df = filtered_source_df.join(
        test_df, on=["Scaled_Image_Path"], how="anti"
    )

    train_df = get_equally_sampled_data(remaining_df, samples_per_class)

    test_data_dict = {
        "type": "test",
        "dir": SAMPLED_TEST_DATA_DIR,
        "parquet": SAMPLED_TEST_DATA_PARQUET,
        "data": test_df,
    }

    training_data_dict = {
        "type": "train",
        "dir": SAMPLED_TRAINING_DATA_DIR,
        "parquet": SAMPLED_TRAINING_DATA_PARQUET,
        "data": train_df,
    }

    data_list = [test_data_dict, training_data_dict]

    for data_dict in data_list:
        data_dir = data_dict["dir"]
        data_parquet = data_dict["parquet"]
        data_df = data_dict["data"]

        data_df = data_df.with_columns(pl.lit(str(data_dir)).alias("target_dir"))

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            if data_parquet.exists():
                data_parquet.unlink(missing_ok=True)
        elif args.force:
            logging.info(f"Removing {data_dir} and parquet file")
            shutil.rmtree(data_dir)
            data_dir.mkdir(parents=True)
            data_parquet.unlink(missing_ok=True)
        elif not args.force:
            logging.error(
                f"Directory {data_dir} already exists. Use -f to remove / recreate it."
            )
            exit(1)

        if not data_dir.exists():
            logging.error(f"Directory {data_dir} does not exist and it should")
            exit(2)

        data_df = parallelize_dataframe(data_df, create_sampled_data, num_cpus)
        data_df = data_df.rename({"Image_Path": "Source_Image_Path"})
        data_df = data_df.rename({"target_dir": "Image_Path"})

        data_df.write_parquet(data_parquet)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
