import multiprocessing as mp
import platform
import shutil
import uuid
from pathlib import Path

import numpy as np
import polars as pl
from platformdirs import user_documents_dir

from utils import *

SAMPLES_PER_CLASS = 110
RANDOM_SEED = 42
TRAIN_PERCENTAGE = 0.8


def get_sampled_data(raw_train_df: pl.DataFrame) -> pl.DataFrame:
    """
    Oversample our data to a median class size of 220 samples if the class has more than SAMPLES_PER_CLASS
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
    shutil.copy(src_image_path, target_image_path)
    #print(f"Copying {src_image_path} to {target_image_path}")
    return str(target_image_path)


def create_sampled_data(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.struct(["target_dir", "ClassId", "Scaled_Image_Path"])
        .map_elements(
            lambda x: duplicate_image(x["target_dir"], x["ClassId"], x["Scaled_Image_Path"]),
        )
        .alias("Sampled_Image_Path")
    )
    return df


def main():
    # Blindly oversample the data to ensure consistent class distribution
    # we can be smarter later on
    # Load the data
    pc = ProjectConfig()
    target_name = "sampled_training_data"
    target_dir = pc.data_root_dir.joinpath(target_name)

    source_data = pc.data_root_dir.joinpath("training_data.parquet")
    sampled_data = get_sampled_data(pl.read_parquet(source_data))
    sampled_data = sampled_data.with_columns(
        pl.lit(str(target_dir)).alias("target_dir")
    )

    target_dir.mkdir(parents=True, exist_ok=True)
    target_parquet = pc.data_root_dir.joinpath(target_name + ".parquet")

    sampled_data = parallelize_dataframe(sampled_data, create_sampled_data, 8)
    sampled_data = sampled_data.drop("target_dir")

    sampled_data.write_parquet(target_parquet)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
