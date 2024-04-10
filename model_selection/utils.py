import itertools
import json
import multiprocessing as mp
import os
import platform
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
from PIL import Image
from platformdirs import user_documents_dir


def parallelize_dataframe(
    df: pl.DataFrame,
    func: Callable[[pl.DataFrame], pl.DataFrame],
    n_cores: int = 4,
) -> pl.DataFrame:
    """
    Enable parallel processing of a dataframe by splitting it by the number of cores
    and then recombining the results.
    """
    rows_per_dataframe = df.height // n_cores
    remainder = df.height % n_cores
    num_rows = [rows_per_dataframe] * (n_cores - 1)
    num_rows.append(rows_per_dataframe + remainder)
    start_pos = [0]
    for n in num_rows:
        start_pos.append(start_pos[-1] + n)
    df_split = []
    for start, rows in zip(start_pos, num_rows):
        df_split.append(df.slice(start, rows))
    func_list = list(itertools.repeat(func, len(df_split)))
    pool_args = list(zip(df_split, func_list))
    pool = mp.Pool(n_cores)
    new_df = pl.concat(pool.map(process_chunk, pool_args))
    pool.close()
    pool.join()
    return new_df


def process_chunk(args: tuple) -> pl.DataFrame:
    """
    Process a chunk of the dataframe
    """
    df, func = args
    new_df = func(df)
    return new_df


def convert_numpy_to_bytesio(image: np.array) -> bytes:
    """
    Save a numpy array to a BytesIO object name is image
    """
    mem_file = BytesIO()
    np.savez_compressed(mem_file, image=image)
    # np.save(mem_file, image)
    return mem_file.getvalue()


def update_path(path: Path, root_dir: Path) -> str:
    return str(root_dir.joinpath(path).resolve())


def read_image(image_path: Path) -> tuple:
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Missing File {image_path}")
        return (0, 0, 0)
    image = Image.open(image_path)
    if image.format == "PNG":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")
    else:
        image = image.convert("RGB")

    image_data = np.asarray(image, dtype=np.float32) / 255.0
    image_height = image_data.shape[0]
    image_width = image_data.shape[1]
    image_resolution = image_height * image_width
    return (
        image_width,
        image_height,
        image_resolution,
    )


def read_image_wrapper(df: pl.DataFrame) -> pl.DataFrame:
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
        pl.col("Image_Path")
        .map_elements(
            lambda x: dict(
                zip(
                    ("Width", "Height", "Resolution"),
                    read_image(x),
                )
            )
        )
        .alias("New_Cols")
    ).unnest("New_Cols")
    return df


class ProjectConfig:
    def __init__(self):
        self.system = platform.system()
        self.user_documents_dir = Path(user_documents_dir())
        self.configuration_file = Path(user_documents_dir()).joinpath(
            "gotmeals_config.json"
        )
        if self.config_exists():
            print("Configuration file found. Using settings from file.")
            self.read_config()
        else:
            print("No configuration file found. Using defaults.")
            if platform.system() == "Windows":
                self.class_root_dir = self.user_documents_dir.joinpath(
                    "01-Berkeley/210"
                )
                self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
                self.data_root_dir = self.class_root_dir.joinpath("data")
            elif platform.system() == "Linux":
                self.class_root_dir = Path("/tf/notebooks")
                self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
                self.data_root_dir = self.class_root_dir.joinpath("data")

    def config_exists(self):
        if not self.configuration_file.exists():
            return False
        else:
            return True

    def read_config(self):
        try:
            with open(self.configuration_file, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            print("Error: JSON decoding failed. Reason:", e)
            config_scuccessful = False
        else:
            class_root_dir = Path(config["class_root_dir"])
            if class_root_dir.exists():
                self.class_root_dir = class_root_dir
                print(f"Class root configured from file: {self.class_root_dir}")
            else:
                if platform.system() == "Windows":
                    self.class_root_dir = self.user_documents_dir.joinpath(
                        "01-Berkeley/210"
                    )
                elif platform.system() == "Linux":
                    self.class_root_dir = Path("/tf/notebooks")
                print(f"Class root not found. Using default: {self.class_root_dir}")
            project_root_dir = Path(config["project_root_dir"])
            if project_root_dir.exists():
                self.project_root_dir = project_root_dir
                print(f"Project root configured from file. {self.project_root_dir}")
            else:
                if platform.system() == "Windows":
                    self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
                elif platform.system() == "Linux":
                    self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
                print(f"Project root not found. Using default: {self.project_root_dir}")
            data_root_dir = Path(config["data_root_dir"])
            if data_root_dir.exists():
                self.data_root_dir = data_root_dir
                print(f"Data root configured from file. {self.data_root_dir}")
            else:
                if platform.system() == "Windows":
                    self.data_root_dir = self.class_root_dir.joinpath("data")
                elif platform.system() == "Linux":
                    self.data_root_dir = self.class_root_dir.joinpath("data")
                print(f"Data root not found. Using default: {self.data_root_dir}")
