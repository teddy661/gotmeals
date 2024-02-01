import csv
import logging
from blake3 import blake3
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

SA_CSV_FILE = Path("sa_models.csv")


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def hash_file(file_path):
    BUF_SIZE = 65536
    bl3 = blake3()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            bl3.update(data)
    return bl3.hexdigest()


def main():
    CURRENT_DIR = Path(__file__).resolve().parent
    MODEL_DIR = CURRENT_DIR.joinpath("../sa_models").resolve()
    BUF_SIZE = 65536

    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        logging.info(f"Directory {MODEL_DIR} created.")

    if SA_CSV_FILE.exists():
        with open(SA_CSV_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["model_name"]
                model_file = MODEL_DIR.joinpath(model_name)
                if model_file.exists():
                    logging.info(f"Model {model_name} found at {model_file}.")
                    model_hash = hash_file(model_file)
                    if model_hash == row["model_hash"]:
                        print(f"Model {model_name} is good to go.")
                else:
                    logging.error(f"Model {model_name} not found at {model_file}.")
                    with TqdmUpTo(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1,
                        desc=row["model_name"],
                    ) as t:
                        urlretrieve(
                            row["model_url"],
                            filename=model_file,
                            reporthook=t.update_to,
                        )
                        t.total = t.n
                    logging.info(
                        f"Model {model_name} downloaded from {row['model_url']}."
                    )
                    model_hash = hash_file(model_file)
                    if model_hash == row["model_hash"]:
                        print(f"Model {model_name} is good to go.")
    else:
        logging.error(f"File {SA_CSV_FILE} does not exist.")
        exit()


if __name__ == "__main__":
    main()
