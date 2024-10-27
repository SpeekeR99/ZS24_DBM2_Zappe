from src.download import dataset_download


def main():
    dataset_download("cblesa/world-of-warcraft-battlegrounds", "data1")
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria", "data2")


if __name__ == "__main__":
    main()
