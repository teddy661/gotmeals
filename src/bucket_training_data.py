import argparse
import logging
import multiprocessing as mp
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import psutil

from utils import *

pc = ProjectConfig()

target_name = "separated_training_data"
TRAINING_DATA_PATH = pc.data_root_dir.joinpath(target_name)
SOURCE_DATA_PATH = pc.data_root_dir.joinpath("training_data_71")

# def rescale_image_for_imagenet(
#    class_id: str, image_path: str, new_image_size: int = 256
# ) -> np.array:
# return


def fix_paths(path: Path, new_root_dir: Path) -> str:
    file_name = path.name
    parent_dir = path.parent.name
    return str(new_root_dir.joinpath(parent_dir).joinpath(file_name).resolve())


#  copy_file(Path(x["local_file_path"]), x["true_class_label"], x["predicted_class_label"], TRAINING_DATA_PATH)
def copy_file(
    src_file: Path, true_label: str, predicted_label: str, dest_dir: Path
) -> str:
    dest_dir = dest_dir.joinpath(true_label)
    if true_label != predicted_label:
        sub_dir = "wrong"
    else:
        sub_dir = "correct"
    dest_dir = dest_dir.joinpath(sub_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir.joinpath(src_file.name)
    shutil.copy(src_file, dest_file)
    return str(dest_file)


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

    common_dataset_path = pc.data_root_dir.joinpath(
        "training_data_71_prediction_results.parquet"
    )
    training_data_parquet_file = pc.data_root_dir.joinpath(target_name + ".parquet")

    if not TRAINING_DATA_PATH.exists():
        TRAINING_DATA_PATH.mkdir(parents=True)
        if training_data_parquet_file.exists():
            training_data_parquet_file.unlink(missing_ok=True)
    elif args.force:
        logging.info(f"Removing {TRAINING_DATA_PATH} and parquet file")
        shutil.rmtree(TRAINING_DATA_PATH)
        training_data_parquet_file.unlink(missing_ok=True)
    elif not args.force:
        logging.error(f"Directory {TRAINING_DATA_PATH} already exists")
        exit(1)

    logging.info(f"Reading {common_dataset_path}")
    df = pl.read_parquet(common_dataset_path)

    # Convert the file paths to local paths just in case
    df = df.with_columns(
        pl.col("file_path")
        .map_elements(
            lambda x: fix_paths(Path(x), SOURCE_DATA_PATH), return_dtype=pl.Utf8
        )
        .alias("local_file_path")
    )

    # Check that the local paths exist just to be sure
    df = df.with_columns(
        pl.col("local_file_path")
        .map_elements(lambda x: Path(x).exists(), return_dtype=pl.Boolean)
        .alias("local_file_exists")
    )
    missing_files = df.filter(pl.col("local_file_exists") == False)
    if missing_files.height > 0:
        logging.error(f"Missing files: {missing_files}")
        exit(1)
    else:
        logging.info(f"All files exist")

    df = df.with_columns(
        pl.struct(["local_file_path", "true_class_labels", "predicted_label"])
        .map_elements(
            lambda x: copy_file(
                Path(x["local_file_path"]),
                x["true_class_labels"],
                x["predicted_label"],
                TRAINING_DATA_PATH,
            ),
            return_dtype=pl.Utf8,
        )
        .alias("final_file_path")
    )
    print(df.head)
    df.write_parquet(training_data_parquet_file, compression="lz4")
    logging.info(f"Writing {training_data_parquet_file}")
    exit(1)

    logging.info(f"Scaling images")

    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8
    df = parallelize_dataframe(df, scale_image_wrapper, num_cpus)

    df.write_parquet(training_data_parquet_file, compression="lz4")
    logging.info(f"Writing {training_data_parquet_file}")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
