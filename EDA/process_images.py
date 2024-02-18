import logging
import multiprocessing as mp
import platform
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from blake3 import blake3
from PIL import Image
from skimage.transform import rescale, rotate

from utils import *

pc = ProjectConfig()
EXTRACTED_COMMON_IMAGES_PATH = pc.data_root_dir.joinpath("extracted_common_images")


def center_crop(image: np.array, target_height: int, target_width: int) -> np.array:
    """
    Perform a center crop on the input image.

    Parameters:
    - image: NumPy array representing the input image.
    - target_height: Desired height of the cropped image.
    - target_width: Desired width of the cropped image.

    Returns:
    - Cropped image.
    """
    height, width = image.shape[:2]

    # Calculate the crop box
    crop_top = max(0, (height - target_height) // 2)
    crop_left = max(0, (width - target_width) // 2)
    crop_bottom = min(height, crop_top + target_height)
    crop_right = min(width, crop_left + target_width)

    # Perform the crop
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right, :]

    return cropped_image


def rescale_image(image: np.ndarray, new_image_size: int = 64) -> tuple:
    """
    Rescale the image to a standard size. Median for our dataset is 35x35.
    Use order = 5 for (Bi-quintic) #Very slow Super high quality result.
    Settle on 64x64 for our standard size after discussion with professor.
    There will be some cropping of the image, but we'll center the crop.
    This function will take an input image for type uint8 or float(64,32)
    and always return an image of dtype float64 which we truncate back to float32
    which is our standard image format due to cvtColor limitations
    """
    scale = new_image_size / min(image.shape[:2])
    image = rescale(image, scale, order=5, anti_aliasing=True, channel_axis=2)
    image = image[
        int(image.shape[0] / 2 - new_image_size / 2) : int(
            image.shape[0] / 2 + new_image_size / 2
        ),
        int(image.shape[1] / 2 - new_image_size / 2) : int(
            image.shape[1] / 2 + new_image_size / 2
        ),
        :,
    ]
    scaled_image_height = image.shape[0]
    scaled_image_width = image.shape[1]
    return (
        scaled_image_width,
        scaled_image_height,
        image,
    )


def rescale_image_for_imagenet(
    class_id: str, image_path: str, new_image_size: int = 256
) -> np.array:
    image_path = Path(image_path)
    image = Image.open(image_path)
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
    cropped_image_hash = blake3(cropped_image.tobytes()).hexdigest()
    image_target_dir = EXTRACTED_COMMON_IMAGES_PATH.joinpath(class_id)
    if not image_target_dir.exists():
        image_target_dir.mkdir(parents=True)
    image_abs_path = image_target_dir.joinpath(cropped_image_hash + ".png")
    cropped_image_pil = Image.fromarray((cropped_image * 255).astype(np.uint8))
    cropped_image_pil.save(image_abs_path)
    return (
        cropped_image.shape[1],
        cropped_image.shape[0],
        cropped_image.shape[0] * cropped_image.shape[1],
        str(image_abs_path),
    )


def scale_image_wrapper(df: pl.DataFrame) -> pl.DataFrame:
    """
    To parallelize the workflow each of the functions previously defined needs
    to be wrapped in a function that takes a dataframe and returns a dataframe.
    This one reads an image from disk and stores it as a flattened list in a column.
    We convert the image to float32 and normalize it to the range of 0-1. cvtColor is which
    we use extensively expects thing in uint8 format. We'll convert back to float32.
    The meta images are png with 4 channels add .convert('RGB') to convert to 3 channels
    doesn't affect the existing jpg
    """
    df = df.with_columns(
        pl.struct(["ClassId", "Image_Path"])
        .map_elements(
            lambda x: dict(
                zip(
                    (
                        "Scaled_Width",
                        "Scaled_Height",
                        "Scaled_Resolution",
                        "Scaled_Image_Path",
                    ),
                    rescale_image_for_imagenet(x["ClassId"], x["Image_Path"]),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    return df


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    common_dataset_path = pc.data_root_dir.joinpath(
        "common_ingredient_images_dataset.parquet"
    )

    if not EXTRACTED_COMMON_IMAGES_PATH.exists():
        EXTRACTED_COMMON_IMAGES_PATH.mkdir(parents=True)
    logging.info(f"Reading {common_dataset_path}")
    df = pl.read_parquet(common_dataset_path)
    logging.info(f"Scaling images")
    df = parallelize_dataframe(df, scale_image_wrapper, 8)

    scaled_image_parquet = pc.data_root_dir.joinpath("extracted_common_images.parquet")
    df.write_parquet(scaled_image_parquet, compression="lz4")
    logging.info(f"Writing {scaled_image_parquet}")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
