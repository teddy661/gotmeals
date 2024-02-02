import pickle
from pathlib import Path

import cv2
import joblib
import numpy as np
import pillow_heif
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

SAM_MODEL_DIR = Path("../sa_models")


def main():
    heic_test_image_1 = Path("../data/sample_data/IMG_0119.HEIC")
    heic_test_image_2 = Path("../data/sample_data/IMG_0118.HEIC")
    png_test_image_1 = Path("../data/sample_data/IMG_0119.PNG")
    png_test_image_2 = Path("../data/sample_data/IMG_0118.PNG")
    image_1_masks_joblib = Path("./masks_img_0119_heic.lzma")

    # 8/10/12 bit HEIF to 8/16 bit PNG using OpenCV
    heif_file_1 = pillow_heif.open_heif(
        heic_test_image_1, convert_hdr_to_8bit=False, bgr_mode=True
    )
    image_1 = np.asarray(heif_file_1)
    if not png_test_image_1.exists():
        cv2.imwrite(str(png_test_image_1), image_1)

    # 8/10/12 bit HEIF to 8/16 bit PNG using OpenCV
    heif_file_2 = pillow_heif.open_heif(
        heic_test_image_2, convert_hdr_to_8bit=False, bgr_mode=True
    )
    image_2 = np.asarray(heif_file_2)
    if not png_test_image_2.exists():
        cv2.imwrite(str(png_test_image_2), image_2)

    sam_checkpoint = SAM_MODEL_DIR.joinpath("sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_model = sam_model_registry[model_type](sam_checkpoint)
    sam_model.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    if image_1_masks_joblib.exists():
        joblib.load(image_1_masks_joblib)
    else:
        # This is very very slow on a 2080Ti with 12GB of VRAM
        # Not as bad on a 4500 with 16GB of VRAM
        masks = mask_generator.generate(image_1)
        joblib.dump(
            masks, image_1_masks_joblib, compress=3, protocol=pickle.HIGHEST_PROTOCOL
        )

    # bbox is XYWH format
    x1 = masks[0]["bbox"][0]
    y1 = masks[0]["bbox"][1]
    x2 = x1 + masks[0]["bbox"][2]
    y2 = y1 + masks[0]["bbox"][3]


if __name__ == "__main__":
    main()
