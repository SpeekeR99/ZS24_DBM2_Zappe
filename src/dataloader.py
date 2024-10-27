import pandas as pd


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

    # Check for missing values
    # print("Missing values:")
    # print(df.isnull().sum())

    return df


def game_map_to_format(game_map):
    maps_10_v_10 = ["Warsong Gulch", "Twin Peaks", "Battle for Gilneas", "Temple of Kotmogu", "Silvershard Mines"]
    maps_15_v_15 = ["Arathi Basin", "Eye of the Storm", "Deepwind Gorge"]
    maps_40_v_40 = ["Alterac Valley", "Isle of Conquest", "The Battle for Gilneas"]

    if game_map in maps_10_v_10:
        return "10v10"
    elif game_map in maps_15_v_15:
        return "15v15"
    elif game_map in maps_40_v_40:
        return "40v40"
    else:
        return "unknown"


def transform_data_to_numeric(df):
    print("Transforming data...")

    # Throw away game_type information -- all are BattleGrounds
    df = df.drop(columns=["game_type"])

    # player_id is float for some reason
    df["player_id"] = df["player_id"].astype(int)

    # Add a column for the faction of the Player and the format of the BattleGround
    df["format"] = df["game_map"].apply(game_map_to_format)

    # Transform start_time and duration to number
    df["start_time"] = pd.to_datetime(df["start_time"]).astype(int)
    # Duration is in format mm:ss, convert it to seconds
    df["duration"] = df["duration"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    # Replace all strings with one hot encoding (race, cls, game_map, format)
    race_mapping = {value: idx for idx, value in enumerate(df["race"].unique())}
    cls_mapping = {value: idx for idx, value in enumerate(df["cls"].unique())}
    game_map_mapping = {value: idx for idx, value in enumerate(df["game_map"].unique())}
    format_mapping = {value: idx for idx, value in enumerate(df["format"].unique())}

    # Index to the one hot encoding
    df["race"] = df["race"].map(race_mapping)
    df["cls"] = df["cls"].map(cls_mapping)
    df["game_map"] = df["game_map"].map(game_map_mapping)
    df["format"] = df["format"].map(format_mapping)

    # One hot encoding
    # df["race"] = df["race"].apply(lambda x: [1 if x == key else 0 for key in race_mapping.keys()])
    # df["cls"] = df["cls"].apply(lambda x: [1 if x == key else 0 for key in cls_mapping.keys()])
    # df["game_map"] = df["game_map"].apply(lambda x: [1 if x == key else 0 for key in game_map_mapping.keys()])
    # df["format"] = df["format"].apply(lambda x: [1 if x == key else 0 for key in format_mapping.keys()])

    return df, {"race": race_mapping, "cls": cls_mapping, "game_map": game_map_mapping, "format": format_mapping}
