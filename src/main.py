from src.download import dataset_download
from src.dataloader import load_data


def main():
    # Download the data
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria", force_download=False)

    # Load the data
    df = load_data("data/games.csv", "data/results.csv")

    # Check for missing values
    print("Missing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
