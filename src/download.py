import os
import shutil
import kagglehub

# Constants
DEFAULT_KAGGLE_DOWNLOAD_CACHE_PATH = "~/.cache/kagglehub"
DEFAULT_CUSTOM_DATA_PATH = "data"


def dataset_download(dataset_name, force_download=False):
    """
    Download the dataset from Kaggle
    :param dataset_name: Name of the dataset
    :param force_download: If True, the data will be downloaded again
    :return: Path to the downloaded data
    """
    default_download_path = os.path.expanduser(DEFAULT_KAGGLE_DOWNLOAD_CACHE_PATH)
    custom_path = os.path.abspath(DEFAULT_CUSTOM_DATA_PATH)

    if not os.path.exists(custom_path):
        os.makedirs(custom_path)
    else:  # Already cached
        if force_download:
            shutil.rmtree(custom_path)
            os.makedirs(custom_path)
        elif os.listdir(custom_path):  # Truly cached
            return

    print("Downloading data...")
    kagglehub.dataset_download(dataset_name)

    downloaded_data_path = os.path.join(default_download_path, "datasets", dataset_name.split("/")[0], dataset_name.split("/")[1], "versions")
    downloaded_data_path = os.path.join(downloaded_data_path, os.listdir(downloaded_data_path)[0])
    for file in os.listdir(downloaded_data_path):
        source = os.path.join(downloaded_data_path, file)
        destination = custom_path
        shutil.move(source, destination)
        print(f"Moved {source} to {destination}")

    shutil.rmtree(default_download_path)
