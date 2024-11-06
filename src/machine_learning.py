import os
import numpy as np
import pickle
import torch
from sklearn.model_selection import train_test_split
from dataloader import game_map_to_format

"""
Approach 1:
SubModel 1) Predict the killing blows, deaths, damage, healing, etc. of a player based on his history
SubModel 2) Predict the outcome of a match based on the map and the players in the match
SubModel 3) Predict the duration of a match based on the map and the players in the match
Model: Combine the three submodels to predict the outcome of a match based on the map and the players in the match
- - - - - - -
Approach 2:
End-to-end model: Predict the outcome of a match based on the map and the players history in the match
"""

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default hyperparameters for training loop
LR = 0.001
BATCH_SIZE = 128
EPOCHS = 10


def log(message):
    """
    Log the message to the console and to the log file
    :param message: Message to log
    """
    print(message)
    with open("models/log.txt", "a") as fp:
        fp.write(message + "\n")


# ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ #
# |            Approach 1)                                                                                           | #
# └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ #


class SubModel1(torch.nn.Module):
    """
    Regressor: input last 10 games of a player (15 features per game); output next game of the player (15 features)
    Simple model that picks up on the trend of the player and predicts the next game stats
    """
    def __init__(self):
        """
        Constructor
        """
        super(SubModel1, self).__init__()

        self.relu = torch.nn.ReLU()

        self.proj1 = torch.nn.Linear(150, 1024)
        self.proj2 = torch.nn.Linear(1024, 256)
        self.head = torch.nn.Linear(256, 15)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Predicted next game stats
        """
        proj = self.proj1(x)
        proj = self.relu(proj)

        proj = self.proj2(proj)
        proj = self.relu(proj)

        final = self.head(proj)
        return final


class RandomBaseline1(SubModel1):
    """
    Random baseline model for SubModel1
    """
    def __init__(self):
        """
        Constructor
        """
        super(RandomBaseline1, self).__init__()

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Random normally distributed numbers (player stats act normally)
        """
        # Normally distributed random numbers (because player stats are normally distributed -- hopefully)
        return torch.randn((x.shape[0], 15))


class SubModel2(torch.nn.Module):
    """
    Classificator: input map and players in the match (predicted by SubModel1); output win/loss of Team 1
    """
    def __init__(self, num_players=10):
        """
        Constructor
        :param num_players: Number of players in a team
        num_players is important because we have to train 3 different models (one for each format)
        """
        super(SubModel2, self).__init__()

        self.relu = torch.nn.ReLU()

        per_player_features = 15
        num_teams = 2
        self.proj1 = torch.nn.Linear(num_players * per_player_features * num_teams + 1, 2048)  # + 1 for the map
        self.proj2 = torch.nn.Linear(2048, 1024)
        self.proj3 = torch.nn.Linear(1024, 256)
        self.head = torch.nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Predicted win/loss of Team 1
        """
        proj = self.proj1(x)
        proj = self.relu(proj)

        proj = self.proj2(proj)
        proj = self.relu(proj)

        proj = self.proj3(proj)
        proj = self.relu(proj)

        final = self.head(proj)
        return final


class RandomBaseline2(SubModel2):
    """
    Random baseline model for SubModel2
    """
    def __init__(self):
        """
        Constructor
        """
        super(RandomBaseline2, self).__init__()

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Random win/loss (50% chance of winning) (uniform distribution here)
        """
        return torch.rand((x.shape[0], 1))


class SubModel3(torch.nn.Module):
    """
    Regressor: input map and players in the match (predicted by SubModel1); output duration of the game
    """
    def __init__(self, num_players=10):
        """
        Constructor
        :param num_players: Number of players in a team
        num_players is important because we have to train 3 different models (one for each format)
        """
        super(SubModel3, self).__init__()

        self.relu = torch.nn.ReLU()

        per_player_features = 15
        num_teams = 2
        self.proj1 = torch.nn.Linear(num_players * per_player_features * num_teams + 1, 2048)  # + 1 for the map
        self.proj2 = torch.nn.Linear(2048, 1024)
        self.proj3 = torch.nn.Linear(1024, 256)
        self.head = torch.nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Predicted duration of the game
        """
        proj = self.proj1(x)
        proj = self.relu(proj)

        proj = self.proj2(proj)
        proj = self.relu(proj)

        proj = self.proj3(proj)
        proj = self.relu(proj)

        final = self.head(proj)
        return final


class RandomBaseline3(SubModel3):
    """
    Random baseline model for SubModel3
    """
    def __init__(self):
        """
        Constructor
        """
        super(RandomBaseline3, self).__init__()

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Random duration (normally distributed)
        """
        # Duration of the matches is normally distributed -- hopefully
        return torch.randn((x.shape[0], 1))


class ModelOfModels(torch.nn.Module):
    """
    Model that combines the three submodels to predict the outcome of a match based on the map and the players in the match
    """
    def __init__(self, sub_model_1, sub_model_2, sub_model_3, num_players=10):
        """
        Constructor
        :param sub_model_1: Sub model 1
        :param sub_model_2: Sub model 2
        :param sub_model_3: Sub model 3
        :param num_players: Number of players in a team
        Number of player is important for the choice of correct sub model 2 and 3
        """
        super(ModelOfModels, self).__init__()

        self.sub_model_1 = sub_model_1
        self.sub_model_2 = sub_model_2
        self.sub_model_3 = sub_model_3

        self.num_players = num_players
        self.per_player_features = 15
        self.num_teams = 2
        self.history = 10
        self.sub_model_1_input = self.num_players * self.per_player_features * self.history * self.num_teams
        self.sub_model_2_input = self.num_players * self.per_player_features * self.num_teams + 1  # + 1 for the map
        self.sub_model_3_input = self.num_players * self.per_player_features * self.num_teams + 1  # + 1 for the map

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Predicted win/loss of Team 1 and duration of the game
        """
        # SubModel1
        x = x.reshape(-1, (self.sub_model_1_input + 1))
        player_stats = x[:, :-1]
        player_stats = player_stats.reshape(-1, self.per_player_features * self.history)

        player_stats_predicted = self.sub_model_1(player_stats).reshape(-1, self.num_players * self.per_player_features * self.num_teams)

        # SubModel2
        last_column = x[:, -1].reshape(-1, 1)
        input_2_3 = torch.cat((player_stats_predicted, last_column), dim=1)
        win_loss = self.sub_model_2(input_2_3)

        # SubModel3
        duration = self.sub_model_3(input_2_3)

        # Concat the outputs
        result = torch.cat((win_loss, duration), dim=1)

        return result


class RandomBaselineModelOfModels(ModelOfModels):
    """
    Random baseline model for ModelOfModels
    """
    def __init__(self, sub_model_1, sub_model_2, sub_model_3, num_players=10):
        """
        Constructor
        :param sub_model_1: Sub model 1
        :param sub_model_2: Sub model 2
        :param sub_model_3: Sub model 3
        :param num_players: Number of players in a team

        """
        super(RandomBaselineModelOfModels, self).__init__(sub_model_1, sub_model_2, sub_model_3, num_players)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Random win/loss of Team 1 and random duration
        """
        # Random baseline for SubModel2
        win_loss = torch.rand((x.shape[0], 1))

        # Random baseline for SubModel3
        duration = torch.randn((x.shape[0], 1))

        # Concat the outputs
        result = torch.tensor([win_loss, duration], dtype=torch.float32).to(device)

        return result


def train_model(model, train_data, train_target, test_data, test_target, loss_function, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS, acc=False, e2e=False):
    """
    Train the model
    :param model: Model to train
    :param train_data: Training data
    :param train_target: Training target
    :param test_data: Testing data
    :param test_target: Testing target
    :param loss_function: Loss function
    :param lr: Learning rate
    :param batch_size: Batch size
    :param epochs: Number of epochs
    :param acc: Calculate accuracy
    :param e2e: End-to-end model flag
    :return: Trained model
    """
    # Init
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    log("Training model...")
    for epoch in range(epochs):
        running_test_loss = 0.0
        test_samples = 0
        running_loss = 0.0
        samples = 0
        train_acc = 0
        test_acc = 0

        for i, batch in enumerate(range(0, train_data.shape[0], batch_size)):
            data = train_data[batch:batch + batch_size]
            target = train_target[batch:batch + batch_size]

            optimizer.zero_grad()

            output = model(data)

            # E2E model has different output format
            if e2e:
                win_loss = output[:, 0].reshape(-1, 1)
                duration = output[:, 1].reshape(-1, 1)
                output = torch.cat((torch.round(torch.sigmoid(win_loss)), duration), dim=1)

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            # Test the model
            if i % batch_size == 0:
                model.eval()

                with torch.no_grad():
                    test_output = model(test_data)

                    # E2E model has different output format
                    if e2e:
                        test_win_loss = test_output[:, 0].reshape(-1, 1)
                        test_duration = test_output[:, 1].reshape(-1, 1)
                        test_output = torch.cat((torch.round(torch.sigmoid(test_win_loss)), test_duration), dim=1)

                    test_loss = loss_function(test_output, test_target)

                    running_test_loss += test_loss.item()
                    test_samples += 1

                    # Basically "if model is classificator"
                    if acc:
                        # E2E model has different output format
                        if e2e:
                            test_acc += ((test_output[:, 0] == test_target[:, 0]).sum().item() + (test_output[:, 1] == test_target[:, 1]).sum().item()) / test_target.shape[0]
                        # Basically "else model is regressor"
                        else:
                            test_output = torch.round(torch.sigmoid(test_output))
                            test_acc += (test_output == test_target).sum().item() / test_target.shape[0]

                model.train()

            running_loss += loss.item()
            samples += 1

            # Basically "if model is classificator"
            if acc:
                # E2E model has different output format
                if e2e:
                    train_acc += ((output[:, 0] == target[:, 0]).sum().item() + (output[:, 1] == target[:, 1]).sum().item()) / target.shape[0]
                # Basically "else model is regressor"
                else:
                    output = torch.round(torch.sigmoid(output))
                    train_acc += (output == target).sum().item() / target.shape[0]

        loss = running_loss / samples
        test_loss = running_test_loss / test_samples

        # Basically "if model is classificator"
        if acc:
            train_acc /= samples
            test_acc /= test_samples
            log(f"Epoch {epoch}, Train Loss: {loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")
        # Basically "else model is regressor"
        else:
            log(f"Epoch {epoch}, Train Loss: {loss}, Test Loss: {test_loss}")

    log("Training finished")
    model.eval()
    return model


def test_model_against_baseline(model, baseline, test_data, test_target, loss_function, acc=False, e2e=False):
    """
    Test the model against a random baseline
    :param model: Model to test
    :param baseline: Baseline model to compare to
    :param test_data: Testing data
    :param test_target: Testing target
    :param loss_function: Loss function
    :param acc: Calculate accuracy
    :param e2e: End-to-end model flag
    """
    # Turn off grads
    model.eval()
    baseline.eval()

    model_output = model(test_data)
    baseline_output = baseline(test_data)

    model_loss = loss_function(model_output, test_target)
    baseline_loss = loss_function(baseline_output, test_target)

    # E2E model has different output format
    if e2e:
        model_win_loss = model_output[:, 0].reshape(-1, 1)
        model_duration = model_output[:, 1].reshape(-1, 1)
        model_output = torch.cat((torch.round(torch.sigmoid(model_win_loss)), model_duration), dim=1)

        baseline_win_loss = baseline_output[:, 0].reshape(-1, 1)
        baseline_duration = baseline_output[:, 1].reshape(-1, 1)
        baseline_output = torch.cat((torch.round(torch.sigmoid(baseline_win_loss)), baseline_duration), dim=1)

    log(f"Model loss: {model_loss.item()}, Random Baseline loss: {baseline_loss.item()}")

    # Basically "if model is classificator"
    if acc:
        # E2E model has different output format
        if e2e:
            model_acc = ((model_output[:, 0] == test_target[:, 0]).sum().item() + (model_output[:, 1] == test_target[:, 1]).sum().item()) / test_target.shape[0]
            baseline_acc = ((baseline_output[:, 0] == test_target[:, 0]).sum().item() + (baseline_output[:, 1] == test_target[:, 1]).sum().item()) / test_target.shape[0]
        else:
            model_output = torch.round(torch.sigmoid(model_output))
            baseline_output = torch.round(torch.sigmoid(baseline_output))

            model_acc = (model_output == test_target).sum().item() / test_target.shape[0]
            baseline_acc = (baseline_output == test_target).sum().item() / test_target.shape[0]
        log(f"Model accuracy: {model_acc}, Random Baseline accuracy: {baseline_acc}")


def get_mock_db_for_sub_model_1(df, how_many_last_games=10):
    """
    This function serves as a mock database for player history stats
    This would be replaced by a real database with real-time updates of player stats in the real world
    :param df: DataFrame with player stats
    :param how_many_last_games: How many last games to take for each player
    """
    # Get the data and the header
    header = df.columns.values.tolist()
    data = df.to_numpy()

    # Group the data by player_id
    player_dict = {}
    for line in data:
        player_id_idx = header.index("player_id")
        player_id = line[player_id_idx]

        if player_id not in player_dict:
            player_dict[player_id] = []
        player_dict[player_id].append(line)

    # Take last 10 games of each player
    for key in player_dict:
        player_dict[key] = np.array(player_dict[key][:how_many_last_games])
        # If the player has less than 10 games, make it 10 by averaging his games and padding with the average
        while player_dict[key].shape[0] < how_many_last_games:
            player_dict[key] = np.vstack((player_dict[key], player_dict[key].mean(axis=0)))

    return player_dict


def get_player_stats_from_mock_db(mock_db, player_id):
    """
    Get the player stats from the mock database
    This could be replaced by a real database query in the real world
    :param mock_db: Mock database
    :param player_id: Player ID
    :return: Player stats
    """
    # Normal player case (if he has less than 10 games, the mock_db already solved that)
    if player_id in mock_db:
        return mock_db[player_id]
    # New player case
    else:
        # Here is my idea how to solve this:
        # 1) Return the average stats of all players in their last 10 games
        # 2) Return some lower percentile, assuming that the new player is not as good as the average player
        # For simplicity, I chose the first option
        return np.array([mock_db[key].mean(axis=0) for key in mock_db])


def sub_model_1_data_preprocess(df):
    """
    Preprocess the data for SubModel1
    :param df: DataFrame with player stats
    :return: Train and test data and target, normalization dictionary
    """
    log("Sub Model 1 data preprocessing...")

    # Get the data in the format for the model
    how_many_last_games = 11  # 10 for training, 1 for testing
    player_dict = get_mock_db_for_sub_model_1(df, how_many_last_games=how_many_last_games)

    # Split the data into data and target
    train_data = []
    train_target = []

    for key in player_dict:
        train_data.append(player_dict[key][:(how_many_last_games - 1)])
        train_target.append(player_dict[key][-1])

    # Convert lists to numpy arrays
    train_data = np.array(train_data, dtype=np.float32)
    train_data = train_data.reshape(-1, train_data.shape[1] * train_data.shape[2])
    train_target = np.array(train_target, dtype=np.float32)

    # Normalize the data
    data_mean = train_data.mean(axis=0)
    data_std = train_data.std(axis=0)
    target_mean = train_target.mean(axis=0)
    target_std = train_target.std(axis=0)
    train_data = (train_data - data_mean) / data_std
    train_target = (train_target - target_mean) / target_std

    # Split the data into train and test
    train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.2)

    # Convert the data to PyTorch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target = torch.tensor(test_target, dtype=torch.float32).to(device)

    normalization_dict = {
        "data_mean": data_mean,
        "data_std": data_std,
        "target_mean": target_mean,
        "target_std": target_std
    }

    log("Sub Model 1 data preprocessing finished")

    return train_data, train_target, test_data, test_target, normalization_dict


def sub_model_1(df, force_retrain=False):
    """
    Construct and train SubModel1 (+ cache it)
    :param df: DataFrame with player stats
    :param force_retrain: Force retrain the model
    :return: Trained model, normalization dictionary
    """
    path_to_model = "models/sub_model_1.pth"
    path_to_norm_dict = "models/sub_model_1_norm_dict.pkl"
    log("Sub Model 1 started")

    if os.path.exists(path_to_model) and not force_retrain:
        log("Loading pretrained model from cache...")
        # Load the model
        model = SubModel1()
        model.load_state_dict(torch.load(path_to_model, weights_only=True))

        if os.path.exists(path_to_norm_dict):
            norm_dict = pickle.load(open(path_to_norm_dict, "rb"))
        else:
            _, _, _, _, norm_dict = sub_model_1_data_preprocess(df)
            pickle.dump(norm_dict, open(path_to_norm_dict, "wb"))
    else:
        log("Creating model from scratch...")
        # Preprocess the data
        train_data, train_target, test_data, test_target, norm_dict = sub_model_1_data_preprocess(df)

        # Train the model
        model = SubModel1()
        loss_function = torch.nn.MSELoss()
        model = train_model(model, train_data, train_target, test_data, test_target, loss_function)

        # Test the model against a random baseline
        random_baseline = RandomBaseline1()
        test_model_against_baseline(model, random_baseline, test_data, test_target, loss_function)

        # Save the model
        torch.save(model.state_dict(), path_to_model)
        pickle.dump(norm_dict, open(path_to_norm_dict, "wb"))

    log("Sub Model 1 finished")

    return model, norm_dict


def sub_model_2_and_3_data_preprocess(df, mappings, model_1, norm_dict_1, target):
    """
    Preprocess the data for SubModel2 and SubModel3
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param model_1: SubModel1
    :param norm_dict_1: Normalization dictionary from SubModel1
    :param target: Target for the model (that is the only difference between SubModel2 and SubModel3)
    :return: Train and test data and target, normalization dictionary
    """
    log("Sub Model 2 and 3 data preprocessing...")

    # Pre-compute necessary indices
    header = df.columns.values.tolist()
    match_id_idx = header.index("match_id")
    player_id_idx = header.index("player_id")
    map_idx = header.index("game_map")
    format_idx = header.index("format")
    winner_idx = header.index("winner")
    duration_idx = header.index("duration")

    # Mock database and initial data
    mock_db = get_mock_db_for_sub_model_1(df, how_many_last_games=10)
    data = df.to_numpy()

    # For each match, get the players and the map
    match_dict = {}
    for line in data:
        match_id = line[match_id_idx]

        if match_id not in match_dict:
            match_dict[match_id] = ([line[map_idx], line[format_idx], line[duration_idx]], [], [])
        if line[winner_idx]:
            match_dict[match_id][1].append(get_player_stats_from_mock_db(mock_db, line[player_id_idx]))
        else:
            match_dict[match_id][2].append(get_player_stats_from_mock_db(mock_db, line[player_id_idx]))

    # Pad the data to have n players in each team (n depends on the format)
    padding_player = np.zeros((10, 15))  # Hard coded shapes, because I could not find a better way, sorry
    format_mapping = mappings["format"]
    reverse_format_mapping = {v: k for k, v in format_mapping.items()}
    for key in match_dict:
        format = reverse_format_mapping[match_dict[key][0][1]]
        expected_size = int(format.split("v")[0])
        match_dict[key][0].append(str(expected_size))

        while len(match_dict[key][1]) < expected_size:
            match_dict[key][1].append(padding_player)
        while len(match_dict[key][2]) < expected_size:
            match_dict[key][2].append(padding_player)

        # What the heck is wrong with the data, there were 3 bg's with 11 players on one team in a 10v10 format
        while len(match_dict[key][1]) > expected_size:
            match_dict[key][1].pop()
        while len(match_dict[key][2]) > expected_size:
            match_dict[key][2].pop()

    # Split the data into data and target (be careful about the format)
    train_data = {"10": [], "15": [], "40": []}
    train_target = {"10": [], "15": [], "40": []}

    for i, key in enumerate(match_dict):
        index = match_dict[key][0][3]  # expected_size from earlier (but string)

        players_win = np.array(match_dict[key][1], dtype=np.float32)
        players_win = players_win.reshape(-1, players_win.shape[1] * players_win.shape[2])
        players_win = (players_win - norm_dict_1["data_mean"]) / norm_dict_1["data_std"]

        players_loss = np.array(match_dict[key][2], dtype=np.float32)
        players_loss = players_loss.reshape(-1, players_loss.shape[1] * players_loss.shape[2])
        players_loss = (players_loss - norm_dict_1["data_mean"]) / norm_dict_1["data_std"]

        # Randomly shuffle the team order, because the order should not matter for the winning condition
        if np.random.rand() < 0.5:
            players = np.vstack((players_win, players_loss))
            if target == "winner":
                train_target[index].append([1])
            if target == "duration":
                train_target[index].append([match_dict[key][0][2]])
        else:
            players = np.vstack((players_loss, players_win))
            if target == "winner":
                train_target[index].append([0])
            if target == "duration":
                train_target[index].append([match_dict[key][0][2]])

        output = list(model_1(torch.tensor(players).to(device)).detach().cpu().numpy().reshape(-1))
        output.append(match_dict[key][0][0])

        train_data[index].append(np.array(output))

    # Convert lists to numpy arrays
    for key in train_data:
        train_data[key] = np.array(train_data[key], dtype=np.float32)
        train_target[key] = np.array(train_target[key], dtype=np.float32)

    # No need for normalization, because the SubModel1 returns normalized players and the map is a number between 0, 10
    # Unless duration is in the target, that needs to be normalized
    means = {}
    stds = {}
    if target == "duration":
        for key in train_target:
            duration_mean = train_target[key].mean()
            duration_std = train_target[key].std()
            train_target[key] = (train_target[key] - duration_mean) / duration_std
            means[key] = duration_mean
            stds[key] = duration_std

    # Split the data into train and test
    test_data = {}
    test_target = {}
    for key in train_data:
        train_data[key], temp_data, train_target[key], temp_target = train_test_split(train_data[key], train_target[key], test_size=0.2)
        test_data[key] = temp_data
        test_target[key] = temp_target

    # Convert the data to PyTorch tensors
    for key in train_data:
        train_data[key] = torch.tensor(train_data[key], dtype=torch.float32).to(device)
        train_target[key] = torch.tensor(train_target[key], dtype=torch.float32).to(device)
        test_data[key] = torch.tensor(test_data[key], dtype=torch.float32).to(device)
        test_target[key] = torch.tensor(test_target[key], dtype=torch.float32).to(device)

    log("Sub Model 2 and 3 data preprocessing finished")

    normalization_dict = {
        "means": means,
        "stds": stds
    }

    return train_data, train_target, test_data, test_target, normalization_dict


def sub_model_2(df, mappings, model_1, norm_dict_1, force_retrain=False):
    """
    Construct and train SubModel2 (+ cache it)
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param model_1: SubModel1
    :param norm_dict_1: Normalization dictionary from SubModel1
    :param force_retrain: Force retrain the model
    :return: Trained models
    """
    base_path_to_model = "models/sub_model_2"
    path_to_model = {"10": f"{base_path_to_model}_10.pth", "15": f"{base_path_to_model}_15.pth", "40": f"{base_path_to_model}_40.pth"}
    log("Sub Model 2 started")

    if os.path.exists(path_to_model["10"]) and os.path.exists(path_to_model["15"]) and os.path.exists(path_to_model["40"]) and not force_retrain:
        log("Loading pretrained models from cache...")
        # Load the models
        models = {
            "10": SubModel2(num_players=10),
            "15": SubModel2(num_players=15),
            "40": SubModel2(num_players=40)
        }
        models["10"].load_state_dict(torch.load(path_to_model["10"], weights_only=True))
        models["15"].load_state_dict(torch.load(path_to_model["15"], weights_only=True))
        models["40"].load_state_dict(torch.load(path_to_model["40"], weights_only=True))
    else:
        log("Creating models from scratch...")
        # Transform the data
        train_data, train_target, test_data, test_target, _ = sub_model_2_and_3_data_preprocess(df, mappings, model_1, norm_dict_1, "winner")

        # Train the models
        models = {
            "10": SubModel2(num_players=10),
            "15": SubModel2(num_players=15),
            "40": SubModel2(num_players=40)
        }
        loss_function = torch.nn.BCEWithLogitsLoss()
        random_baseline = RandomBaseline2()

        for key in models:
            # Train the model
            models[key] = train_model(models[key], train_data[key], train_target[key], test_data[key], test_target[key], loss_function, acc=True)

            # Test the model against a random baseline
            test_model_against_baseline(models[key], random_baseline, test_data[key], test_target[key], loss_function, acc=True)

            # Save the model
            torch.save(models[key].state_dict(), path_to_model[key])

    log("Sub Model 2 finished")

    return models


def sub_model_3(df, mappings, model_1, norm_dict_1, force_retrain=False):
    """
    Construct and train SubModel3 (+ cache it)
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param model_1: SubModel1
    :param norm_dict_1: Normalization dictionary from SubModel1
    :param force_retrain: Force retrain the model
    :return: Trained models, normalization dictionary
    """
    base_path_to_model = "models/sub_model_3"
    path_to_norm_dict = "models/sub_model_3_norm_dict.pkl"
    path_to_model = {"10": f"{base_path_to_model}_10.pth", "15": f"{base_path_to_model}_15.pth", "40": f"{base_path_to_model}_40.pth"}
    log("Sub Model 3 started")

    if os.path.exists(path_to_model["10"]) and os.path.exists(path_to_model["15"]) and os.path.exists(path_to_model["40"]) and not force_retrain:
        log("Loading pretrained models from cache...")
        # Load the models
        models = {
            "10": SubModel3(num_players=10),
            "15": SubModel3(num_players=15),
            "40": SubModel3(num_players=40)
        }
        models["10"].load_state_dict(torch.load(path_to_model["10"], weights_only=True))
        models["15"].load_state_dict(torch.load(path_to_model["15"], weights_only=True))
        models["40"].load_state_dict(torch.load(path_to_model["40"], weights_only=True))

        if os.path.exists(path_to_norm_dict):
            norm_dict = pickle.load(open(path_to_norm_dict, "rb"))
        else:
            _, _, _, _, norm_dict = sub_model_2_and_3_data_preprocess(df, mappings, model_1, norm_dict_1, "duration")
            pickle.dump(norm_dict, open(path_to_norm_dict, "wb"))
    else:
        log("Creating models from scratch...")
        # Transform the data
        train_data, train_target, test_data, test_target, norm_dict = sub_model_2_and_3_data_preprocess(df, mappings, model_1, norm_dict_1, "duration")

        # Train the models
        models = {
            "10": SubModel3(num_players=10),
            "15": SubModel3(num_players=15),
            "40": SubModel3(num_players=40)
        }
        loss_function = torch.nn.MSELoss()
        random_baseline = RandomBaseline3()

        for key in models:
            # Train the model
            models[key] = train_model(models[key], train_data[key], train_target[key], test_data[key], test_target[key], loss_function)

            # Test the model against a random baseline
            test_model_against_baseline(models[key], random_baseline, test_data[key], test_target[key], loss_function)

            # Save the model
            torch.save(models[key].state_dict(), path_to_model[key])

        # Save the normalization dictionary
        pickle.dump(norm_dict, open(path_to_norm_dict, "wb"))

    log("Sub Model 3 finished")

    return models, norm_dict


def model_of_models(df, mappings, force_retrain=False):
    """
    Construct and train ModelOfModels
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param force_retrain: Force retrain the model
    :return: Trained models, normalization dictionaries
    """
    log("Model of Models started")

    if not os.path.exists("models"):
        os.makedirs("models")

    # Create the submodels
    model_1, norm_dict_1 = sub_model_1(df, force_retrain=force_retrain)
    models_2 = sub_model_2(df, mappings, model_1, norm_dict_1, force_retrain=force_retrain)
    models_3, norm_dict_3 = sub_model_3(df, mappings, model_1, norm_dict_1, force_retrain=force_retrain)

    models = {
        "10": ModelOfModels(model_1, models_2["10"], models_3["10"], num_players=10),
        "15": ModelOfModels(model_1, models_2["15"], models_3["15"], num_players=15),
        "40": ModelOfModels(model_1, models_2["40"], models_3["40"], num_players=40)
    }

    log("Model of Models finished")

    return models, norm_dict_1, norm_dict_3


# ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ #
# |            Approach 2)                                                                                           | #
# └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ #


class EndToEndModel(torch.nn.Module):
    """
    End-to-end model: Predict the outcome of a match based on the map and the players history in the match
    Input: last 10 games of each player in the match (15 features per player); output: win/loss and duration of game
    """
    def __init__(self, num_players=10):
        """
        Constructor
        :param num_players: Number of players in a team
        """
        super(EndToEndModel, self).__init__()

        self.num_players = num_players
        self.per_player_features = 15
        self.history = 10
        self.num_teams = 2

        # input_size is either 3001 or 4501 or 12001
        self.input_size = self.num_players * self.per_player_features * self.history * self.num_teams + 1  # + 1 for the map

        self.relu = torch.nn.ReLU()
        self.proj1 = torch.nn.Linear(self.input_size, 2048)
        self.proj2 = torch.nn.Linear(2048, 1024)
        self.proj3 = torch.nn.Linear(1024, 256)
        self.head1 = torch.nn.Linear(256, 1)
        self.head2 = torch.nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Output data
        """
        x = x.reshape(-1, self.input_size)

        proj = self.proj1(x)
        proj = self.relu(proj)

        proj = self.proj2(proj)
        proj = self.relu(proj)

        proj = self.proj3(proj)
        proj = self.relu(proj)

        win_loss = self.head1(proj)
        duration = self.head2(proj)

        if win_loss.shape[0] == 1:
            win_loss = win_loss.unsqueeze(0)
            duration = duration.unsqueeze(0)

        # Concat the outputs
        result = torch.cat((win_loss, duration), dim=1)
        return result


class RandomBaselineEndToEndModel(EndToEndModel):
    """
    Random baseline model for EndToEndModel
    """
    def __init__(self, num_players=10):
        """
        Constructor
        :param num_players: Number of players in a team
        """
        super(RandomBaselineEndToEndModel, self).__init__(num_players)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Random win/loss of Team 1 and random duration
        """
        win_loss = torch.rand((x.shape[0], 1))
        duration = torch.randn((x.shape[0], 1))
        # Concat the outputs
        result = torch.cat((win_loss, duration), dim=1).to(device)
        return result


def end_to_end_data_preprocess(df, mappings, player_norm_dict):
    """
    End-to-end Model data preprocessing
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param player_norm_dict: Normalization dictionary from SubModel1 / preprocessing of e2e data (same thing)
    :return: Train and test data and target, normalization dictionary
    """
    log("End-to-end Model data preprocessing...")

    # Pre-compute necessary indices
    header = df.columns.values.tolist()
    match_id_idx = header.index("match_id")
    player_id_idx = header.index("player_id")
    map_idx = header.index("game_map")
    format_idx = header.index("format")
    winner_idx = header.index("winner")
    duration_idx = header.index("duration")

    # Mock database and initial data
    mock_db = get_mock_db_for_sub_model_1(df, how_many_last_games=10)
    data = df.to_numpy()

    # For each match, get the players and the map
    match_dict = {}
    for line in data:
        match_id = line[match_id_idx]

        if match_id not in match_dict:
            match_dict[match_id] = ([line[map_idx], line[format_idx], line[duration_idx]], [], [])
        if line[winner_idx]:
            match_dict[match_id][1].append(get_player_stats_from_mock_db(mock_db, line[player_id_idx]))
        else:
            match_dict[match_id][2].append(get_player_stats_from_mock_db(mock_db, line[player_id_idx]))

    # Pad the data to have n players in each team (n depends on the format)
    padding_player = np.zeros((10, 15))  # Hard coded shapes, because I could not find a better way, sorry
    format_mapping = mappings["format"]
    reverse_format_mapping = {v: k for k, v in format_mapping.items()}
    for key in match_dict:
        format = reverse_format_mapping[match_dict[key][0][1]]
        expected_size = int(format.split("v")[0])
        match_dict[key][0].append(str(expected_size))

        while len(match_dict[key][1]) < expected_size:
            match_dict[key][1].append(padding_player)
        while len(match_dict[key][2]) < expected_size:
            match_dict[key][2].append(padding_player)

        # What the heck is wrong with the data, there were 3 bg's with 11 players on one team in a 10v10 format
        while len(match_dict[key][1]) > expected_size:
            match_dict[key][1].pop()
        while len(match_dict[key][2]) > expected_size:
            match_dict[key][2].pop()

    # Split the data into data and target (be careful about the format)
    train_data = {"10": [], "15": [], "40": []}
    train_target = {"10": [[], []], "15": [[], []], "40": [[], []]}

    for i, key in enumerate(match_dict):
        index = match_dict[key][0][3]  # expected_size from earlier (but string)

        players_win = np.array(match_dict[key][1], dtype=np.float32)
        players_win = players_win.reshape(-1, players_win.shape[1] * players_win.shape[2])
        players_win = (players_win - player_norm_dict["data_mean"]) / player_norm_dict["data_std"]

        players_loss = np.array(match_dict[key][2], dtype=np.float32)
        players_loss = players_loss.reshape(-1, players_loss.shape[1] * players_loss.shape[2])
        players_loss = (players_loss - player_norm_dict["data_mean"]) / player_norm_dict["data_std"]

        # Randomly shuffle the team order, because the order should not matter for the winning condition
        if np.random.rand() < 0.5:
            players = np.vstack((players_win, players_loss))
            train_target[index][0].append([1])
            train_target[index][1].append([match_dict[key][0][2]])
        else:
            players = np.vstack((players_loss, players_win))
            train_target[index][0].append([0])
            train_target[index][1].append([match_dict[key][0][2]])

        players = list(players.reshape(-1))
        players.append(match_dict[key][0][0])
        train_data[index].append(np.array(players))

    # Convert lists to numpy arrays
    for key in train_data:
        train_data[key] = np.array(train_data[key], dtype=np.float32)
        train_target[key][0] = np.array(train_target[key][0], dtype=np.float32)
        train_target[key][1] = np.array(train_target[key][1], dtype=np.float32)

    means = {}
    stds = {}
    for key in train_target:
        duration_mean = train_target[key][1].mean()
        duration_std = train_target[key][1].std()
        train_target[key][1] = (train_target[key][1] - duration_mean) / duration_std
        means[key] = duration_mean
        stds[key] = duration_std

    # Zip the target lists
    for key in train_target:
        train_target[key] = np.squeeze(np.array(list(zip(train_target[key][0], train_target[key][1])), dtype=np.float32))

    # Split the data into train and test
    test_data = {}
    test_target = {}
    for key in train_data:
        train_data[key], temp_data, train_target[key], temp_target = train_test_split(train_data[key], train_target[key], test_size=0.2)
        test_data[key] = temp_data
        test_target[key] = temp_target

    # Convert the data to PyTorch tensors
    for key in train_data:
        train_data[key] = torch.tensor(train_data[key], dtype=torch.float32).to(device)
        train_target[key] = torch.tensor(train_target[key], dtype=torch.float32).to(device)
        test_data[key] = torch.tensor(test_data[key], dtype=torch.float32).to(device)
        test_target[key] = torch.tensor(test_target[key], dtype=torch.float32).to(device)

    log("End-to-end Model data preprocessing finished")

    normalization_dict = {
        "means": means,
        "stds": stds
    }

    return train_data, train_target, test_data, test_target, normalization_dict


def combined_loss(predictions, target, classification_loss_fn=torch.nn.BCEWithLogitsLoss(), regression_loss_fn=torch.nn.MSELoss(), alpha=0.5):
    """
    Combined loss function for the End-to-end model
    :param predictions: Predictions
    :param target: Target
    :param classification_loss_fn: Loss function for classification
    :param regression_loss_fn: Loss function for regression
    :param alpha: Weight for the classification loss
    :return: Total loss
    """
    # Separate predictions
    win_loss_pred = predictions[:, 0]  # Prediction for win/loss
    duration_pred = predictions[:, 1]  # Prediction for duration

    # Separate target
    win_loss_target = target[:, 0]  # Target for win/loss
    duration_target = target[:, 1]  # Target for duration

    # Compute individual losses
    loss_classification = classification_loss_fn(win_loss_pred, win_loss_target)
    loss_regression = regression_loss_fn(duration_pred, duration_target)

    # Combine with weights
    total_loss = alpha * loss_classification + (1 - alpha) * loss_regression
    return total_loss


def end_to_end_models(df, mappings, force_retrain=False):
    """
    Construct and train End-to-end Model
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param force_retrain: Force retrain the model
    :return: Trained models, normalization dictionaries
    """
    base_path_to_model = "models/end_to_end_model"
    path_to_model = {"10": f"{base_path_to_model}_10.pth", "15": f"{base_path_to_model}_15.pth", "40": f"{base_path_to_model}_40.pth"}
    path_to_player_norm_dict = "models/end_to_end_model_player_norm_dict.pkl"
    path_to_duration_norm_dict = "models/end_to_end_model_duration_norm_dict.pkl"
    log("End-to-end Model started")

    if not os.path.exists("models"):
        os.makedirs("models")

    if os.path.exists(path_to_model["10"]) and os.path.exists(path_to_model["15"]) and os.path.exists(path_to_model["40"]) and not force_retrain:
        log("Loading pretrained models from cache...")
        # Load the models
        models = {
            "10": EndToEndModel(num_players=10),
            "15": EndToEndModel(num_players=15),
            "40": EndToEndModel(num_players=40)
        }
        models["10"].load_state_dict(torch.load(path_to_model["10"], weights_only=True))
        models["15"].load_state_dict(torch.load(path_to_model["15"], weights_only=True))
        models["40"].load_state_dict(torch.load(path_to_model["40"], weights_only=True))

        if os.path.exists(path_to_player_norm_dict) and os.path.exists(path_to_duration_norm_dict):
            player_norm_dict = pickle.load(open(path_to_player_norm_dict, "rb"))
            duration_norm_dict = pickle.load(open(path_to_duration_norm_dict, "rb"))
        else:
            _, _, _, _, player_norm_dict = sub_model_1_data_preprocess(df)
            _, _, _, _, duration_norm_dict = end_to_end_data_preprocess(df, mappings, player_norm_dict)
            pickle.dump(player_norm_dict, open(path_to_player_norm_dict, "wb"))
            pickle.dump(duration_norm_dict, open(path_to_duration_norm_dict, "wb"))
    else:
        log("Creating models from scratch...")
        # Transform the data
        _, _, _, _, player_norm_dict = sub_model_1_data_preprocess(df)
        train_data, train_target, test_data, test_target, duration_norm_dict = end_to_end_data_preprocess(df, mappings, player_norm_dict)

        # Train the models
        models = {
            "10": EndToEndModel(num_players=10),
            "15": EndToEndModel(num_players=15),
            "40": EndToEndModel(num_players=40)
        }
        loss_function = combined_loss
        random_baseline = RandomBaselineEndToEndModel()

        for key in models:
            # Train the model
            models[key] = train_model(models[key], train_data[key], train_target[key], test_data[key], test_target[key], loss_function, acc=True)

            # Test the model against a random baseline
            test_model_against_baseline(models[key], random_baseline, test_data[key], test_target[key], loss_function, acc=True)

            # Save the model
            torch.save(models[key].state_dict(), path_to_model[key])

        # Save the normalization dictionary
        pickle.dump(player_norm_dict, open(path_to_player_norm_dict, "wb"))
        pickle.dump(duration_norm_dict, open(path_to_duration_norm_dict, "wb"))

    log("End-to-end Model finished")

    return models, player_norm_dict, duration_norm_dict


def test_models_against_each_other(df, mappings):
    """
    Test the approach 1 against the approach 2 (model of models vs end to end models)
    """
    models_1, _, _ = model_of_models(df, mappings)
    models_2, player_norm_dict, _ = end_to_end_models(df, mappings)
    _, _, test_data, test_target, _ = end_to_end_data_preprocess(df, mappings, player_norm_dict)
    loss_function = combined_loss

    # Turn off grads
    for key in models_1:
        model_1 = models_1[key]
        model_2 = models_2[key]

        model_1.eval()
        model_2.eval()

        output_1 = model_1(test_data[key])
        output_2 = model_2(test_data[key])

        loss_1 = loss_function(output_1, test_target[key])
        loss_2 = loss_function(output_2, test_target[key])

        log(f"Approach 1 loss: {loss_1.item()}, Approach 2 loss: {loss_2.item()}")

        model_1_win_loss = output_1[:, 0].reshape(-1, 1)
        model_1_duration = output_1[:, 1].reshape(-1, 1)
        model_1_output = torch.cat((torch.round(torch.sigmoid(model_1_win_loss)), model_1_duration), dim=1)

        model_2_win_loss = output_2[:, 0].reshape(-1, 1)
        model_2_duration = output_2[:, 1].reshape(-1, 1)
        model_2_output = torch.cat((torch.round(torch.sigmoid(model_2_win_loss)), model_2_duration), dim=1)

        model_1_acc = ((model_1_output[:, 0] == test_target[key][:, 0]).sum().item() + (model_1_output[:, 1] == test_target[key][:, 1]).sum().item()) / test_target[key].shape[0]
        model_2_acc = ((model_2_output[:, 0] == test_target[key][:, 0]).sum().item() + (model_2_output[:, 1] == test_target[key][:, 1]).sum().item()) / test_target[key].shape[0]

        log(f"Approach 1 accuracy: {model_1_acc}, Approach 2 accuracy: {model_2_acc}")


# ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐ #
# |            Real usage of models                                                                                  | #
# └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘ #


def transform_teams_and_map(db, mappings, team_1, team_2, match_map, player_norm_dict):
    """
    Transform user input in form of lists of ID's and string for map name into a format that the model can understand
    :param db: Database to take the player stats from
    :param mappings: Mappings for the data
    :param team_1: List of player ID's in team 1
    :param team_2: List of player ID's in team 2
    :param match_map: Name of the map
    :param player_norm_dict: Normalization dictionary from SubModel1 / preprocessing of e2e data (same thing)
    :return: Data in the format that the model can understand
    """
    padding_player = np.zeros((10, 15))  # Hard coded shapes, because I could not find a better way, sorry
    num_players_per_team = int(game_map_to_format(match_map).split("v")[0])

    team_1 = [get_player_stats_from_mock_db(db, player_id) for player_id in team_1]
    while len(team_1) < num_players_per_team:
        team_1.append(padding_player)
    while len(team_1) > num_players_per_team:
        team_1.pop()
    team_1 = np.array(team_1, dtype=np.float32)
    # Normalize the team 1
    team_1 = ((team_1.reshape(-1, team_1.shape[1] * team_1.shape[2]) - player_norm_dict["data_mean"]) / player_norm_dict["data_std"]).reshape(-1)

    team_2 = [get_player_stats_from_mock_db(db, player_id) for player_id in team_2]
    while len(team_2) < num_players_per_team:
        team_2.append(padding_player)
    while len(team_2) > num_players_per_team:
        team_2.pop()
    team_2 = np.array(team_2, dtype=np.float32)
    # Normalize the team 2
    team_2 = ((team_2.reshape(-1, team_2.shape[1] * team_2.shape[2]) - player_norm_dict["data_mean"]) / player_norm_dict["data_std"]).reshape(-1)

    match_map = np.array([mappings["game_map"][match_map]], dtype=np.float32)

    data = np.concatenate((team_1, team_2, match_map)).reshape(-1)
    return data.astype(np.float32)


def use_model(models, duration_norm_dict, data, match_map):
    """
    Use the model to predict the outcome of the match
    :param models: Models for the prediction (3 formats)
    :param duration_norm_dict: Normalization dictionary for the duration
    :param data: Data in the format that the model can understand
    :param match_map: Name of the map
    :return: Win/loss and duration of the match
    """
    num_players_per_team = game_map_to_format(match_map).split("v")[0]

    model = models[num_players_per_team]

    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    output = model(data).detach().cpu().numpy().reshape(-1)

    win_loss = output[0]
    duration = output[1]

    # Win / loss - sigmoid and round
    win_loss = torch.round(torch.sigmoid(torch.tensor(win_loss, dtype=torch.float32))).item()

    # Denormalize the duration
    mean = duration_norm_dict["means"][num_players_per_team]
    std = duration_norm_dict["stds"][num_players_per_team]
    duration = duration * std + mean

    return win_loss, duration


def example_usage(df, mappings, models_approach_1, player_norm_dict_1, duration_norm_dict_1, models_approach_2, player_norm_dict_2, duration_norm_dict_2):
    """
    Example usage of the models
    :param df: DataFrame with player stats
    :param mappings: Mappings for the data
    :param models_approach_1: Models for the first approach
    :param player_norm_dict_1: Player normalization dictionary for the first approach
    :param duration_norm_dict_1: Duration normalization dictionary for the first approach
    :param models_approach_2: Models for the second approach
    :param player_norm_dict_2: Player normalization dictionary for the second approach
    :param duration_norm_dict_2: Duration normalization dictionary for the second approach
    """
    print("Example usage of the models...")
    print("Input data:")

    # Team 1 is 4 known players and 4 completely new players (player_id 1, 2, 3, 4)
    team_1 = df["player_id"].sample(4).to_list()
    team_1.append(1)
    team_1.append(2)
    team_1.append(3)
    team_1.append(4)
    print(f"\tPlayer IDs in Team 1 (Your team): {team_1}")

    # Team 2 is 9 known players
    team_2 = df["player_id"].sample(9).to_list()
    print(f"\tPlayer IDs in Team 2 (opponent team): {team_2}")

    match_map = "Warsong Gulch"
    print(f"\tMap: {match_map}")

    db = get_mock_db_for_sub_model_1(df, how_many_last_games=10)
    # player_norm_dict_1 == player_norm_dict_2 ; but if we decide in the future to use only one model, we won't have two
    data = transform_teams_and_map(db, mappings, team_1, team_2, match_map, player_norm_dict_1)

    print("Prediction of model using approach 1:")
    win_loss, duration = use_model(models_approach_1, duration_norm_dict_1, data, match_map)
    print("Raw output:")
    print(f"\t{win_loss}, {duration}")
    print(f"{'Team 1 (You)' if win_loss else 'Team 2 (opponent team)'} have bigger chance of winning")
    print(f"The BattleGround will be about {duration:.0f} seconds long")

    print("Prediction of model using approach 2:")
    win_loss, duration = use_model(models_approach_2, duration_norm_dict_2, data, match_map)
    print("Raw output:")
    print(f"\t{win_loss}, {duration}")
    print(f"{'Team 1 (You)' if win_loss else 'Team 2 (opponent team)'} have bigger chance of winning")
    print(f"The BattleGround will be about {duration:.0f} seconds long")

    print("Example usage of the models finished")
