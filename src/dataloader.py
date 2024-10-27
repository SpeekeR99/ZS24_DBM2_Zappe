import pandas as pd
import pickle
from src.data_utils import BattleGround, Player, game_map_to_format

CACHE_PATH = "data/cache.pickle"


def load_bg_data(file_path):
    df = pd.read_csv(file_path)
    all_lines = df.shape[0]

    # Filter out all lines that are not relevant for BattleGrounds
    df = df[df["game_type"] == "bg"]

    # Some data in the .csv is labeled as "bg" but is not a BattleGround
    df = df[df["game_map"].apply(game_map_to_format) != "unknown"]
    bg_lines = df.shape[0]

    # Create a list of BattleGrounds (containers)
    battlegrounds = {}
    for idx, row in df.iterrows():
        bg = BattleGround(row["match_id"], row["game_map"], row["start_time"], row["duration"])
        battlegrounds[row["match_id"]] = bg

    print(f"Loaded {bg_lines} BattleGrounds out of {all_lines} PvP records")
    print(f"({bg_lines / all_lines * 100:.2f} % of all PvP records are BattleGrounds)")

    return battlegrounds


def load_player_data(file_path, battlegrounds):
    df = pd.read_csv(file_path)
    all_lines = df.shape[0]

    # Iterate over all the players and add them to their corresponding BattleGround
    counter = 0
    for idx, row in df.iterrows():
        # If player played in a BattleGround add him to the corresponding BattleGround
        if row["match_id"] in battlegrounds:
            # If any player data is missing, skip this player (what is important is the race and class mainly)
            # This is basically missing values cleaning
            if pd.isnull(row["player_id"]) or pd.isnull(row["race"] or pd.isnull(row["cls"])):
                continue

            player = Player(row["player_id"], row["race"], row["cls"], row["winner"], row["killing_blows"], row["deaths"], row["damage"], row["healing"], row["damage_taken"], row["healing_taken"])
            battlegrounds[row["match_id"]].players.append(player)
            counter += 1

    print(f"Loaded {counter} Players from BatlleGrounds out of {all_lines} Players")
    print(f"({counter / all_lines * 100:.2f} % of all Players are Players from BattleGrounds)")

    return battlegrounds


def load_data(bg_file_path, player_file_path, force_reload=False):
    print("Loading data...")

    if not force_reload:
        try:
            with open(CACHE_PATH, "rb") as fp:
                battlegrounds = pickle.load(fp)
                print(f"Loaded {len(battlegrounds)} BattleGrounds from cache")
                return battlegrounds
        except FileNotFoundError:
            pass

    battlegrounds = load_bg_data(bg_file_path)
    battlegrounds = load_player_data(player_file_path, battlegrounds)

    with open(CACHE_PATH, "wb") as fp:
        pickle.dump(battlegrounds, fp)
        print(f"Saved {len(battlegrounds)} BattleGrounds to cache")

    return battlegrounds
