import json
import pickle
import random
import time
from io import BytesIO
from pathlib import Path

import polars as pl
import psutil
import requests
import urllib3
import validators
from blake3 import blake3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from utils import *


def download_images(url_list: list, class_dir: Path) -> list:
    user_agent = {
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    http = urllib3.PoolManager(8, headers=user_agent)
    if not class_dir.exists():
        class_dir.mkdir(parents=True)

    image_paths = []
    for i, url in enumerate(url_list, 1):
        print(f"Original URL:\t{url}")
        url = url.split("&s")[0]
        if not validators.url(url):
            image_path = None
            print(f"Invalid URL {url}")
        else:
            time.sleep(random.uniform(0.01, 0.5))
            response = http.request("GET", url)
            if response.status != 200:
                print(f"Failed to download image {i} from {url}")
            else:
                print(f"Downloaded image {i} from {url}")
                image_data = response.data
                image_hash = blake3(image_data).hexdigest()
                image_path = class_dir.joinpath(f"{image_hash}.jpg")
                with open(image_path, "wb") as f:
                    f.write(image_data)
        path_to_append = str(image_path) if image_path is not None else None
        image_paths.append(path_to_append)
    return image_paths


def main():
    target_directory_name = "image_downloads"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    target_directory = Path(f"image_downloads_{timestr}").resolve()
    if not target_directory.exists():
        target_directory.mkdir(parents=True)

    QUERY_FILE = Path("queries.json").resolve()
    with open(QUERY_FILE, "r") as f:
        queries = json.load(f)

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # chrome_options.add_argument("--disable-extensions")

    query_results = []
    for c in queries:
        class_id = c["ClassId"]
        search_strings = c["SearchStrings"]
        print(f"Processing class     :\t{class_id}")
        driver = webdriver.Chrome(
            service=ChromeService(
                ChromeDriverManager().install(), options=chrome_options
            )
        )
        for search_string in search_strings:
            print(f"\tSearch String:\t{search_string}")
            # Create url variable containing the webpage for a Google image search.
            # url = ("https://www.google.com/search?q={s}&tbm=isch&tbs=sur%3Afc&hl=en&ved=0CAIQpwVqFwoTCKCa1c6s4-oCFQAAAAAdAAAAABAC&biw=1251&bih=568")
            # url = ("https://www.google.com/search?q={s}&tbm=isch&tbs=sur%3Afc&hl=en&bih=568")
            url = "https://www.google.com/search?q={s}&sca_esv=f7010a6fa6b14553&tbm=isch&source=hp&biw=1969&bih=1058&ei=xGniZaLeHJfbkPIPqrCZ-A8&iflsig=ANes7DEAAAAAZeJ31G8Uz-MkGwSj2hY82lWUOb-ewC7m&ved=0ahUKEwii0IjToNSEAxWXLUQIHSpYBv8Q4dUDCAc&uact=5&oq=celery&gs_lp=EgNpbWciBmNlbGVyeTIIEAAYgAQYsQMyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgAQyBRAAGIAEMgUQABiABDIFEAAYgARIkAhQAFjJBnAAeACQAQCYAS-gAYACqgEBNrgBA8gBAPgBAYoCC2d3cy13aXotaW1nmAIGoAKMApgDAJIHATY&sclient=img"
            # url = "https://www.google.com/search?q={s}&tbm=isch"
            # Launch the browser and open the given url in the webdriver.
            driver.get(url.format(s=search_string))
            # Scroll down the body of the web page and load the images.
            SCROLL_PAUSE_TIME = random.uniform(2.0, 3.0)

            # Get scroll height
            last_height = driver.execute_script("return document.body.scrollHeight")

            while True:
                # Scroll down to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait to load page
                time.sleep(SCROLL_PAUSE_TIME)

                # Calculate new scroll height and compare with last scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")

                if new_height == last_height:
                    # break #insert press load more
                    try:
                        element = driver.find_elements(
                            By.CLASS_NAME, "LZ4I"
                        )  # returns list
                        element[0].click()
                    except:
                        break
                last_height = new_height

            # Find the images.
            # imgResults = driver.find_elements(By.XPATH, "//img[contains(@class,'Q4LuWd')]")
            image_links = driver.find_elements(By.CLASS_NAME, "YQ4gaf")

            # Access and store the scr list of image url's.
            src_links = [img.get_attribute("src") for img in image_links]
            src_links = [x for x in src_links if "faviconV2" not in x]
            data_src_links = [img.get_attribute("data-src") for img in image_links]

            # combine and clean the links
            combined_links = src_links + data_src_links
            final_links = list(filter(lambda x: x is not None, combined_links))
            # Retrieve and download the images.
            written_files = download_images(
                final_links, target_directory.joinpath(class_id)
            )
            # OrderedDict([('ClassId', String),
            #              ('ImageId', String),
            #              ('Image_Path', String),
            #              ('Width', Int64),
            #              ('Height', Int64),
            #              ('Resolution', Int64)])
            search_results_df = pl.DataFrame(
                {
                    "URL": final_links,
                    "Image_Path": written_files,
                }
            )
            search_results_df = search_results_df.filter(
                pl.col("Image_Path").is_not_null()
            )  # drop nulls from the image download
            search_results_df = search_results_df.with_columns(
                pl.lit(search_string).alias("SearchString"),
                pl.lit(class_id).alias("ClassId"),
                pl.col("Image_Path")
                .map_elements(lambda x: str(Path(x).name), return_dtype=pl.String)
                .alias("ImageId"),
            )
            query_results.append(search_results_df)
        driver.close()

    all_results = pl.concat(query_results)
    num_cpus = psutil.cpu_count(logical=False)
    if num_cpus > 8:
        num_cpus = 8
    print("Analyzing Images")
    all_results = parallelize_dataframe(all_results, read_image_wrapper, num_cpus)

    result_file_name = f"image_downloads_{timestr}.parquet"
    reult_file_path = target_directory.parent.joinpath(result_file_name)
    all_results.write_parquet(
        reult_file_path, compression="lz4", compression_level=6, statistics=True
    )


if __name__ == "__main__":
    main()
