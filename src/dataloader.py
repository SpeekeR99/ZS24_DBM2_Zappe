import pandas as pd
from src.data_utils import game_map_to_format, race_to_faction


def load_bg_data(file_path):
    df = pd.read_csv(file_path)
    all_lines = df.shape[0]

    # Filter out all lines that are not relevant for BattleGrounds
    df = df[df["game_type"] == "bg"]

    # Some data in the .csv is labeled as "bg" but is not a BattleGround
    df = df[df["game_map"].apply(game_map_to_format) != "unknown"]
    bg_lines = df.shape[0]

    print(f"Loaded {bg_lines} BattleGrounds out of {all_lines} PvP records")
    print(f"({bg_lines / all_lines * 100:.2f} % of all PvP records are BattleGrounds)")

    return df


def load_player_data(file_path, battlegrounds_df):
    df = pd.read_csv(file_path)
    all_lines = df.shape[0]

    # Filter out all lines that are not relevant for BattleGrounds
    df = df[df["match_id"].isin(battlegrounds_df["match_id"].values)]

    # Some players (394 737 of them) have missing values, drop them
    df = df.dropna(subset=["player_id", "race", "cls"])

    bg_lines = df.shape[0]

    print(f"Loaded {bg_lines} Players from BatlleGrounds out of {all_lines} Players")
    print(f"({bg_lines / all_lines * 100:.2f} % of all Players are Players from BattleGrounds)")

    return df


def load_data(bg_file_path, player_file_path):
    print("Loading data...")

    bg_df = load_bg_data(bg_file_path)
    player_df = load_player_data(player_file_path, bg_df)

    # Merge the two DataFrames
    df = pd.merge(player_df, bg_df, on="match_id")

    # Add a column for the faction of the Player and the format of the BattleGround
    df["faction"] = df["race"].apply(race_to_faction)
    df["format"] = df["game_map"].apply(game_map_to_format)

    return df
