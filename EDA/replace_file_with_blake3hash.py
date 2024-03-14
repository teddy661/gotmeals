import multiprocessing as mp
import re
from pathlib import Path

from blake3 import blake3

from utils import *

pc = ProjectConfig()

TARGET_DIR_1 = pc.data_root_dir.joinpath(
    "image_downloads_20240302-132212/image_downloads_20240302-132212"
)
TARGET_DIR_2 = pc.data_root_dir.joinpath(
    "image_downloads_20240303-091315/image_downloads_20240303-091315"
)


def recurse_dir(target_dir: Path):
    pattern = r"^(?!.*\b[0-9a-fA-F]{64}\.jpg$).+\.jpg$"
    if not target_dir.exists():
        print(f"Directory {target_dir} does not exist.")
    else:
        for item in target_dir.iterdir():
            if item.is_file():
                if item.suffix in [".jpg", ".png", ".jpeg", ".gif"]:
                    file_hash = blake3()
                    with open(item, "rb") as f:
                        file_hash.update(f.read())
                    hash_name = file_hash.hexdigest()
                    target_name = hash_name + item.suffix
                    abs_target_name = item.parent.joinpath(target_name)
                    # if re.match(pattern, str(item.name)):
                    #    print(f"NotConverted: {item}")
                    if not abs_target_name.exists():
                        item.rename(abs_target_name)
                    else:
                        print(f"Duplicate: {hash_name}: \t{item}")
                        item.unlink()
            if item.is_dir():
                recurse_dir(item)


def main():
    for target_dir in [TARGET_DIR_1, TARGET_DIR_2]:
        recurse_dir(target_dir)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn")
    main()
