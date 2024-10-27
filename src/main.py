from src.download import dataset_download
from src.dataloader import load_data, transform_data_to_numeric
from src.data_analysis import my_pca


def main():
    # Download the data
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria")

    # Load the data
    df = load_data("data/games.csv", "data/results.csv")

    # Transform strings to one hot encoding, transform dates to numbers, drop unnecessary columns
    df, mappings = transform_data_to_numeric(df)

    # PCA
    components, pca = my_pca(df, plot=True)


if __name__ == "__main__":
    main()
