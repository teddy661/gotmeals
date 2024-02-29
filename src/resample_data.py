import multiprocessing as mp
import platform
from pathlib import Path

import polars as pl
from platformdirs import user_documents_dir

SAMPLES_PER_CLASS = 220
RANDOM_SEED = 42
TRAIN_PERCENTAGE = 0.8


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


def get_sampled_data(raw_train_df: pl.DataFrame) -> pl.DataFrame:
    """
    Samples the dataframe to get the same number of samples per class our smallest class
    has 210 examples so we'll use that as our sample size. We'll also shuffle the dataframe
    :param df: The dataframe to sample
    :return: The sampled dataframe
    """
    pl.set_random_seed(RANDOM_SEED)
    train_equal_sample_df = pl.concat(
        [
            x.sample(SAMPLES_PER_CLASS, with_replacement=True, shuffle=False)
            for x in raw_train_df.partition_by("ClassId")
        ]
    )
    return train_equal_sample_df.sample(shuffle=True, fraction=1)


def main():
    # Blindly oversample the data to ensure consistent class distribution
    # Load the data
    pc = ProjectConfig()
    source_data = pc.data_root_dir.joinpath("extracted_common_images.parquet")
    sampled_data = get_sampled_data(pl.read_parquet(source_data))
    target_dir = pc.data_root_dir.joinpath("sampled_data")
    if not target_dir.exists():
        target_dir.mkdir()


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
