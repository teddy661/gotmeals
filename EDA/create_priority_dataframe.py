from pathlib import Path

import polars as pl

top_100_path = Path("top_100_sorted_ingredient_counts.csv")
counts_path = Path("counts_of_ingredients_from_original_dataset.csv")
top_100_df = pl.read_csv(top_100_path)
counts_df = pl.read_csv(counts_path)

top_100_df = top_100_df.rename({"0": "ClassId", "1": "NumRecipes"})
top_100_df = top_100_df.select(pl.col("ClassId"), pl.col("NumRecipes"))
joined_df = top_100_df.join(counts_df, on="ClassId", how="left")
joined_df.sort("NumRecipes", descending=True).write_csv("priority_datasets.csv")
