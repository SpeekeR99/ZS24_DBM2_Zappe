from src.download import dataset_download
from src.dataloader import load_data, transform_data_to_numeric
from src.data_analysis import my_pca
from src.answers import answer_1, answer_2, asnwer_3, answer_4


def main():
    """
    1) Is Paladin, Hunter, Warrior and Death Knight more likely to win than other classes?
    2) Is Human more likely to win than other races?
    3) Is having more healers in the team beneficial?
    4) Are players more active during the weekends?
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    5) Classificator; Input = Here is my character; Output = Did you make a good choice when creating your character?
    6) Regression; Input = Here is my last game; Output = Score of how good you played
    """
    # Download the data
    dataset_download("sosperec/massive-world-of-warcraft-pvp-mists-of-pandaria")

    # Load the data
    df = load_data("data/games.csv", "data/results.csv")

    # Transform strings to one hot encoding, transform dates to numbers, drop unnecessary columns
    df, mappings = transform_data_to_numeric(df)

    # PCA
    # components, pca = my_pca(df, plot=True)

    # Answer the questions
    # answer_1(df, mappings)
    # answer_2(df, mappings)
    # asnwer_3(df, mappings)
    answer_4(df)


if __name__ == "__main__":
    main()
