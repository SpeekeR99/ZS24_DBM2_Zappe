import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
import pandas as pd
import numpy as np


def plot_win_rate(win_rate, win_rate_of_what, x_label, add_count=True):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    width = win_rate["count"] / win_rate["count"].max()
    # Normalize the width to <0.10; 0.75> so the extremes are not so extreme
    lower, upper = 0.10, 0.75
    width = lower + upper * (width - width.min()) / (width.max() - width.min())
    ax.bar(win_rate.index, win_rate["mean"], width=width, color="green")
    for idx, value in enumerate(win_rate["mean"]):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")

    # Plot the distribution based on the count
    if add_count:
        # Normalize the count to range of <0; win_rate[mean].max()>
        range_norm = (win_rate["count"] / win_rate["count"].max()) * win_rate["mean"].max() * 0.75
        spl = make_interp_spline(range(len(win_rate.index)), range_norm, k=3)
        x_smooth = np.linspace(0, len(win_rate.index) - 1, 300)
        y_smooth = spl(x_smooth)

        ax.plot(x_smooth, y_smooth, color="blue", label="Count")

    ax.set_xticks(range(len(win_rate.index)))
    ax.set_xticklabels(win_rate.index, rotation=22.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Win rate")
    ax.set_title(f"Win rate of {win_rate_of_what}")

    # Create legend
    bar_patch = Line2D([0], [0], color="green", label="Win rate")
    line_patch = Line2D([0], [0], color="blue", label="Count")
    if add_count:
        ax.legend(handles=[bar_patch, line_patch], loc="best")
    else:
        ax.legend(handles=[bar_patch], loc="best")

    plt.grid()
    plt.savefig(f"img/win_rate_{win_rate_of_what}.svg")
    plt.show()


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

    # Plot the win rate
    plot_win_rate(win_rate, "classes", "Class", add_count=False)


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

    # Plot the win rate
    plot_win_rate(win_rate, "races", "Race", add_count=False)


def asnwer_3(df, mappings):
    """
    3) Is having more healers in the team beneficial?
    """
    class_mapping = mappings["cls"]
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
    win_rate = win_rate.sort_index(ascending=True)

    # Plot the win rate
    plot_win_rate(win_rate, "healers", "Is Healer", add_count=False)

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
        win_rate_df = win_rate_df.sort_index(ascending=True)

        # Plot the win rate
        plot_win_rate(win_rate_df, f"{format_name}_healers", "Number of Healers")


def answer_4(df):
    """
    4) Are players more active during the weekends?
    """
    # Count number of matches for each day
    df.loc[:, "day"] = pd.to_datetime(df["start_time"], unit="ns").dt.date
    df.loc[:, "number_of_matches_that_day"] = df.groupby("day")["match_id"].transform("count")
    df.loc[:, "day_of_week"] = pd.to_datetime(df["start_time"], unit="ns").dt.dayofweek

    # boxplot function does all this for me
    # # Calculate the average number of matches for each day of the week
    # number_of_matches = {"mean": {}, "std": {}, "count": {}}
    #
    # for day in df["day_of_week"].unique():
    #     number_of_matches["mean"][day] = df[df["day_of_week"] == day]["number_of_matches_that_day"].mean()
    #     number_of_matches["std"][day] = df[df["day_of_week"] == day]["number_of_matches_that_day"].std()
    #     number_of_matches["count"][day] = df[df["day_of_week"] == day]["number_of_matches_that_day"].count()
    #
    # # Sort and print
    # number_of_matches = pd.DataFrame(number_of_matches)
    # days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    # number_of_matches.loc[:, "day_name"] = [days[day] for day in number_of_matches.index]
    # number_of_matches = number_of_matches.sort_index(ascending=True)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Plot the number of matches
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # I will use boxplot here, because standard deviation finally has some sense (unlike with win rate)
    ax.boxplot(
        [df[df["day_of_week"] == day]["number_of_matches_that_day"] for day in range(len(days))],
        labels=days,
        patch_artist=True,
        boxprops=dict(facecolor="green"),
        medianprops=dict(color="yellow", linewidth=2),
        meanline=True,
        showmeans=True,
        meanprops=dict(color="red", linewidth=2),
    )

    # Make legend for the boxplot
    custom_lines = [Line2D([0], [0], color="red", lw=2),
                    Line2D([0], [0], color="yellow", lw=2)]
    ax.legend(custom_lines, ["Mean", "Median"])

    ax.set_xlabel("Day of the week")
    ax.set_ylabel("Number of matches")
    ax.set_title("Number of matches per day of the week")

    plt.grid()
    plt.savefig("img/number_of_matches_per_day.svg")
    plt.show()
