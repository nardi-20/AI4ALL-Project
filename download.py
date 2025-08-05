
import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def download_and_rename_kaggle_dataset(dataset_name, target_filename, download_path="data"):
    os.makedirs(download_path, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset: {dataset_name}")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)

    # Find the first CSV file and rename it to the target name
    for file in os.listdir(download_path):
        if file.endswith(".csv") and file != target_filename:
            if os.path.exists(os.path.join(download_path, target_filename)):
                os.remove(os.path.join(download_path, target_filename))
            original_path = os.path.join(download_path, file)
            new_path = os.path.join(download_path, target_filename)
            os.rename(original_path, new_path)
            print(f"Renamed {file} to {target_filename}")
            break

if __name__ == "__main__":
    dataset = "kalilurrahman/apple-stock-data-live-and-latest-from-ipo-date"  # Replace with your dataset
    download_and_rename_kaggle_dataset(dataset, target_filename="AAPL.csv")
