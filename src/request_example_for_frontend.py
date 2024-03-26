import json
import logging
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from pillow_heif import register_heif_opener
from skimage.transform import rescale

PROTOCOL = "https"
HOST = "edbrown.mids255.com"
PORT = 443

Image.MAX_IMAGE_PIXELS = 110000000

endpoint = f"{PROTOCOL}://{HOST}:{PORT}/predict"

register_heif_opener()

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


def rescale_image_for_imagenet(image_path: str, new_image_size: int = 256) -> np.array:
    """
    Reads a file from the OS with PIL and center crops it to 256,256 then rescale
    it to 224,224 for use with the ImageNet pre-trained models. Should now work with HEIC
    images with the addition of pillow_heif.
    """
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
    bio = BytesIO()
    Image.fromarray((cropped_image * 255).astype(np.uint8)).save(bio, format="PNG")
    return bio


def main():
    debug = True
    file_to_predict = Path("celery.png")
    if not file_to_predict.exists():
        raise FileNotFoundError(f"File {file_to_predict} not found")

    bio = rescale_image_for_imagenet(file_to_predict)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(bio.getvalue())
        f.seek(0)
        files = {"file": f}
        headers = {"accept": "application/json"}
        response = requests.post(endpoint, files=files, headers=headers)

    print(response.status_code)
    print(json.dumps(response.json()))


if __name__ == "__main__":
    main()
