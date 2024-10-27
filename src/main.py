from src.download import dataset_download
from src.dataloader import load_data


def main():
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria", force_download=False)

    battlegrounds = load_data("data/games.csv", "data/results.csv", force_reload=False)


if __name__ == "__main__":
    main()
