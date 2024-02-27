import ast
import logging
import multiprocessing as mp
import platform
from pathlib import Path

import numpy as np
import polars as pl
import psutil
from platformdirs import user_documents_dir

from utils import ProjectConfig, parallelize_dataframe


def clean_text(text: str) -> str:
    """
    Clean the text by removing any non-alphabetic characters and converting to lower case
    """
    return " ".join([word.lower() for word in text.split() if word.isalpha()])


def main():
    pc = ProjectConfig()

    rnlg_data_dir = pc.data_root_dir.joinpath("RecipeDatabase")
    rnlg_csv_file = rnlg_data_dir.joinpath("recipe_dataset_cleaned_v3.csv")
    rnlg_parquet_file = pc.data_root_dir.joinpath("RecipeNLG_dataset.parquet")
    # Load the RecipeNLG dataset
    rnlg_df = pl.read_csv(rnlg_csv_file)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    num_cpus = psutil.cpu_count(logical=False)
    ## Fix the '' column name to id
    rnlg_df.drop_in_place("NER")
    rnlg_df = rnlg_df.rename({"cleaned_NER": "NER"})

    ## Fix the columns into lists from strings
    column_names_convert_to_list = ["ingredients", "directions", "NER"]
    for col_name in column_names_convert_to_list:
        logging.info(f"Converting {col_name} to list")
        rnlg_df = rnlg_df.with_columns(
            pl.col(col_name)
            .map_elements(lambda x: ast.literal_eval(x))
            .alias(col_name),
        )

    ## add a column for the length of the list
    for col_name in ["NER", "directions", "ingredients"]:
        logging.info(f"Adding Counts:\t{col_name}")
        rnlg_df = rnlg_df.with_columns(
            pl.col(col_name)
            .map_elements(lambda lst: len(lst))
            .alias(col_name + "_len"),
        )

    logging.info(f"Writing parquet:\t{rnlg_parquet_file}")
    rnlg_df.write_parquet(
        rnlg_parquet_file, compression="zstd", compression_level=9, statistics=True
    )


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
