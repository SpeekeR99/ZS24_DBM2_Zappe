from src.download import dataset_download
from battleground import *


def main():
    dataset_download("cblesa/world-of-warcraft-battlegrounds", "data1")
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria", "data2")

    with open("data/data2/games.csv", "r") as file:
        lines = file.readlines()
        lines = lines[1:]
        total_lines = len(lines)
        print(total_lines)

        # Filter out all lines that are relevant for BattleGrounds
        lines = [line for line in lines if line.split(",")[2] == "bg"]
        bg_lines = len(lines)
        print(bg_lines)

        # Some data in the .csv is labeled as "bg" but is not a BattleGround
        lines = [line for line in lines if game_map_to_format(line.split(",")[1]) != "unknown"]
        true_bg_lines = len(lines)
        print(true_bg_lines)


if __name__ == "__main__":
    main()
