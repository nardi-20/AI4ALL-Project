import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_file(dataset_name, filename, download_path="data"):
    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset_name}")

    # Download dataset as zip
    zip_path = os.path.join(download_path, f"{dataset_name.replace('/', '_')}.zip")
    api.dataset_download_files(dataset_name, path=download_path, unzip=False)

    print(f"Downloaded zip to {zip_path}")

    # Extract only the requested file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if filename in zip_ref.namelist():
            print(f"Extracting {filename}")
            zip_ref.extract(filename, path=download_path)
        else:
            print(f"File {filename} not found in dataset.")

    # Optional: rename extracted file
    original_file_path = os.path.join(download_path, filename)
    target_file_path = os.path.join(download_path, "AAPL.csv")  # or whatever name you want
    if os.path.exists(original_file_path):
        os.rename(original_file_path, target_file_path)
        print(f"Renamed {filename} to AAPL.csv")

if __name__ == "__main__":
    dataset = "nikhil1e9/netflix-stock-price"
    file_to_extract = "APPLE_daily.csv"  # exact filename inside dataset zip
    download_and_extract_file(dataset, file_to_extract)
