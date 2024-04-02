import argparse
import logging
import multiprocessing as mp
import platform
import shutil
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from blake3 import blake3
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from skimage.transform import rescale, rotate

from utils import *

pc = ProjectConfig()
Image.MAX_IMAGE_PIXELS = 110000000
target_name = "training_data_71_prediction_results.parquet"
TRAINING_DATA_PATH = pc.data_root_dir.joinpath(target_name)


def rescale_image_for_imagenet(
    class_id: str, image_path: str, new_image_size: int = 256
) -> np.array:
    image_path = Path(image_path)
    if not image_path.exists():
        logging.error(f"ERROR: Missing Image: {image_path}")
        return (0, 0, 0, "")
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError:
        logging.error(f"ERROR: Unidentified Image: {image_path}")
        return (0, 0, 0, "")
    if image.format == "PNG":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    image_data = np.asarray(image, dtype=np.float64) / 255.0
    _, _, rs_image = rescale_image(image_data, new_image_size=new_image_size)
    cropped_image = center_crop(rs_image, 224, 224)
    image_target_dir = TRAINING_DATA_PATH.joinpath(class_id)
    if not image_target_dir.exists():
        image_target_dir.mkdir(parents=True)

    with BytesIO() as bio:
        Image.fromarray((cropped_image * 255).astype(np.uint8)).save(bio, format="PNG")
        cropped_image_hash = blake3(bio.getvalue()).hexdigest()
        image_abs_path = image_target_dir.joinpath(cropped_image_hash + ".png")
        with open(image_abs_path, "wb") as f:
            f.write(bio.getvalue())
    return (
        cropped_image.shape[1],
        cropped_image.shape[0],
        cropped_image.shape[0] * cropped_image.shape[1],
        str(image_abs_path),
    )





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
        "merged_dataset_corrected_classes.parquet"
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
    # Certain datasets have huge images that clog up a few threads
    # shuffle the dataframe to hopefully distribute the load better
    # and avoid a few processes getting stuck on huge images
    df = df.sample(fraction=1, seed=142, with_replacement=False, shuffle=True)
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
