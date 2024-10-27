import pandas as pd


def answer_1(df, mappings):
    """
    1) Is Paladin, Hunter, Warrior and Death Knight more likely to win than other classes?
    """
    class_mapping = mappings["cls"]
    class_mapping_reversed = {v: k for k, v in class_mapping.items()}

    # Calculate win rate of classes
    classes = df["cls"].unique()
    win_rate = {"mean": {}, "std": {}, "count": {}}

    for cls in classes:
        class_name = class_mapping_reversed[cls]
        win_rate["mean"][class_name] = df[df["cls"] == cls]["winner"].mean()
        win_rate["std"][class_name] = df[df["cls"] == cls]["winner"].std()
        win_rate["count"][class_name] = df[df["cls"] == cls]["winner"].count()

    # Sort and print
    win_rate = pd.DataFrame(win_rate)
    win_rate = win_rate.sort_values(by="mean", ascending=False)
    print(win_rate)


def answer_2(df, mappings):
    """
    2) Is Human more likely to win than other races?
    """
    race_mapping = mappings["race"]
    race_mapping_reversed = {v: k for k, v in race_mapping.items()}

    # Calculate win rate of races
    races = df["race"].unique()
    win_rate = {"mean": {}, "std": {}, "count": {}}

    for race in races:
        race_name = race_mapping_reversed[race]
        win_rate["mean"][race_name] = df[df["race"] == race]["winner"].mean()
        win_rate["std"][race_name] = df[df["race"] == race]["winner"].std()
        win_rate["count"][race_name] = df[df["race"] == race]["winner"].count()

    # Sort and print
    win_rate = pd.DataFrame(win_rate)
    win_rate = win_rate.sort_values(by="mean", ascending=False)
    print(win_rate)


def asnwer_3(df, mappings):
    """
    3) Is having more healers in the team beneficial?
    """
    class_mapping = mappings["cls"]
    class_mapping_reversed = {v: k for k, v in class_mapping.items()}
    format_mapping = mappings["format"]
    format_mapping_reversed = {v: k for k, v in format_mapping.items()}

    # Following classes can be healers
    healer_classes = ["priest", "shaman", "paladin", "druid", "monk"]
    df.loc[:, "healer"] = False  # By default no one is a healer
    # I defined a healer as a player has more healing than 10 * damage
    df.loc[:, "healer"] = (df["cls"].isin([class_mapping[cls] for cls in healer_classes])) & (df["healing"] > 2 * df["damage"])

    # Sanity check
    # print(df["healer"].value_counts())
    # False    3128510
    # True      611837
    # sum      3740347 (everyone is either a healer or not)
    # also the game tries to put 1 healer, 1 tank and 3 dps in a team of 5 and 611837/3128510 = 0.196 (almost a fifth)

    # Calculate win rate of healers
    win_rate = {"mean": {}, "std": {}, "count": {}}

    for healer in [True, False]:
        win_rate["mean"][healer] = df[df["healer"] == healer]["winner"].mean()
        win_rate["std"][healer] = df[df["healer"] == healer]["winner"].std()
        win_rate["count"][healer] = df[df["healer"] == healer]["winner"].count()

    # Sort and print
    win_rate = pd.DataFrame(win_rate)
    win_rate = win_rate.sort_values(by="mean", ascending=False)
    print(win_rate)

    # First solve healer counts for each match_id
    temp = df.groupby("match_id")["healer"].sum()
    df.loc[:, "healer_count"] = df["match_id"].map(temp)

    win_rate = {}
    # For each format (10v10, 15v15, 40v40) calculate win rate for each healer count
    for group in df["format"].unique():
        format_name = format_mapping_reversed[group]
        win_rate[format_name] = {"mean": {}, "std": {}, "count": {}}

        # For each possible healer number in the match of the given format
        for healer_count in df["healer_count"].unique():
            # It is possible that 20 healers (40v40) were not in 10v10, duh
            if df[(df["format"] == group) & (df["healer_count"] == healer_count)]["winner"].count() == 0:
                continue

            win_rate[format_name]["mean"][healer_count] = df[(df["format"] == group) & (df["healer_count"] == healer_count)]["winner"].mean()
            win_rate[format_name]["std"][healer_count] = df[(df["format"] == group) & (df["healer_count"] == healer_count)]["winner"].std()
            win_rate[format_name]["count"][healer_count] = df[(df["format"] == group) & (df["healer_count"] == healer_count)]["winner"].count()

    # Sort and print
    for group in df["format"].unique():
        format_name = format_mapping_reversed[group]
        win_rate_df = pd.DataFrame(win_rate[format_name])
        win_rate_df = win_rate_df.sort_values(by="mean", ascending=False)
        print(f"Format: {format_name}")
        print(win_rate_df)
