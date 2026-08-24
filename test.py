from src.utils import download_and_extract_zip
import os

index_dir = r".\corpus\textbooks\index\ncbi\MedCPT-Article-Encoder"

url = (
    "https://myuva-my.sharepoint.com/:u:/g/personal/"
    "hhu4zu_virginia_edu/"
    "EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ"
    "?download=1"
)

download_and_extract_zip(
    url,
    os.path.join(index_dir, "embedding.zip"),
    index_dir
)