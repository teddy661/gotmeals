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

from utils import ProjectConfig, convert_numpy_to_bytesio, parallelize_dataframe


def load_image(image_data: bytes) -> np.array:
    """
    Load an image from a byte stream.

    Parameters:
    - image: Byte stream representing the image.

    Returns:
    - NumPy array representing the image.
    """
    t_image = np.load(BytesIO(image_data))
    image = t_image["image"]
    return image


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


def rescale_image(image: bytes, new_image_size: int = 64) -> tuple:
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
    image: np.ndarray, new_image_size: int = 256
) -> np.array:
    _, _, rs_image = rescale_image(image, new_image_size=new_image_size)
    cropped_image = center_crop(rs_image, 224, 224)
    return (
        cropped_image.shape[1],
        cropped_image.shape[0],
        cropped_image,
    )


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    pc = ProjectConfig()
    common_dataset_path = pc.data_root_dir.joinpath(
        "common_ingredient_images_dataset.parquet"
    )
    extracted_common_images_path = pc.data_root_dir.joinpath("extracted_common_images")
    if not extracted_common_images_path.exists():
        extracted_common_images_path.mkdir(parents=True)
    df = pl.read_parquet(common_dataset_path, n_rows=100)
    _, _, new_image = rescale_image_for_imagenet(load_image(df["Image_Data"][0]))
    image_name = blake3(new_image.tobytes()).hexdigest()
    print(type(new_image))
    print(new_image.shape)
    print(hash)
    # logging.info(f"Converting {col_name} to list")


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
