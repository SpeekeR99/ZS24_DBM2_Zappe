from src.download import dataset_download
from src.dataloader import load_data, transform_data_to_numeric


def main():
    # Download the data
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria", force_download=False)

    # Load the data
    df = load_data("data/games.csv", "data/results.csv")

    # Transform strings to one hot encoding, transform dates to numbers, drop unnecessary columns
    df, mappings = transform_data_to_numeric(df)
    print(df.iloc[0])


if __name__ == "__main__":
    main()
