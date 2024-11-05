from src.download import dataset_download
from src.dataloader import load_data, transform_data_to_numeric
from src.data_analysis import correlation, my_pca
from src.statistics import answer_1, answer_2, asnwer_3, answer_4
from machine_learning import model_of_models, end_to_end_models, example_usage


def main():
    """
    Statistical questions:
    1) Is Paladin, Hunter, Warrior and Death Knight more likely to win than other classes?
    2) Is Human more likely to win than other races?
    3) Is having more healers in the team beneficial?
    4) Are players more active during the weekends?
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Machine learning:
    SubModel 1) Predict the killing blows, deaths, damage, healing, etc. of a player based on his history
    SubModel 2) Predict the outcome of a match based on the map and the players in the match
    SubModel 3) Predict the duration of a match based on the map and the players in the match
    Model: Combine the three submodels to predict the outcome of a match based on the map and the players in the match
    """
    # Download the data
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria")

    # Load the data
    df = load_data("data/games.csv", "data/results.csv")

    # Transform strings to one hot encoding, transform dates to numbers, drop unnecessary columns
    df, mappings = transform_data_to_numeric(df)

    # Correlation
    # correlation(df)

    # PCA
    # components, pca = my_pca(df)

    # Answer the questions
    # answer_1(df, mappings)
    # answer_2(df, mappings)
    # asnwer_3(df, mappings)
    # answer_4(df)

    # Machine Learning
    models_approach_1, player_norm_dict_1, duration_norm_dict_1 = model_of_models(df, mappings)
    models_approach_2, player_norm_dict_2, duration_norm_dict_2 = end_to_end_models(df, mappings)

    # Real usage of the models
    example_usage(df, mappings, models_approach_1, player_norm_dict_1, duration_norm_dict_1, models_approach_2, player_norm_dict_2, duration_norm_dict_2)


if __name__ == "__main__":
    main()
