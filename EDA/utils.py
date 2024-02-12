import itertools
import multiprocessing as mp
import os
import platform
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
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
    Save a numpy array to a BytesIO object
    """
    mem_file = BytesIO()
    np.savez_compressed(mem_file, image=image)
    # np.save(mem_file, image)
    return mem_file.getvalue()


class ProjectConfig:
    def __init__(self):
        self.system = platform.system()
        self.user_documents_dir = Path(user_documents_dir())
        if platform.system() == "Windows":
            self.class_root_dir = self.user_documents_dir.joinpath("01-Berkeley/210")
            self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
            self.data_root_dir = self.class_root_dir.joinpath("data")
        elif platform.system() == "Linux":
            self.class_root_dir = Path("/tf/notebooks")
            self.project_root_dir = self.class_root_dir.joinpath("gotmeals")
            self.data_root_dir = self.class_root_dir.joinpath("data")
