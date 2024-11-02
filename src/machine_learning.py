import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SubModel1(torch.nn.Module):
    """
    Regressor: input last 10 games of a player (15 features per game); output next game of the player (15 features)
    Simple model that picks up on the trend of the player and predicts the next game stats
    """
    def __init__(self):
        super(SubModel1, self).__init__()

        self.relu = torch.nn.ReLU()

        self.proj1 = torch.nn.Linear(150, 1024)
        self.proj2 = torch.nn.Linear(1024, 256)
        self.head = torch.nn.Linear(256, 15)

    def forward(self, x):
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
        super(RandomBaseline1, self).__init__()

    def forward(self, x):
        # Normally distributed random numbers (because player stats are normally distributed -- hopefully)
        return torch.randn((x.shape[0], 15))


class SubModel2(torch.nn.Module):
    """
    Classificator: input map and players in the match (predicted by SubModel1); output win/loss of Team 1
    """
    def __init__(self, num_players=10):
        super(SubModel2, self).__init__()

        self.relu = torch.nn.ReLU()

        per_player_features = 15
        num_teams = 2
        self.proj1 = torch.nn.Linear(num_players * per_player_features * num_teams + 1, 2048)  # + 1 for the map
        self.proj2 = torch.nn.Linear(2048, 1024)
        self.proj3 = torch.nn.Linear(1024, 256)
        self.head = torch.nn.Linear(256, 1)

    def forward(self, x):
        proj = self.proj1(x)
        proj = self.relu(proj)

        proj = self.proj2(proj)
        proj = self.relu(proj)

        proj = self.proj3(proj)
        proj = self.relu(proj)

        final = self.head(proj)
        final = torch.round(torch.sigmoid(final))
        return final


class RandomBaseline2(SubModel2):
    """
    Random baseline model for SubModel2
    """
    def __init__(self):
        super(RandomBaseline2, self).__init__()

    def forward(self, x):
        # Uniformly distributed random integer numbers -- either 0 or 1
        return torch.randint(0, 2, (x.shape[0], 1)).float()


def train_model(model, train_data, train_target, test_data, test_target, loss_function, lr=0.001, batch_size=128, epochs=10, acc=False):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training model...")
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

            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            if i % batch_size == 0:
                model.eval()

                with torch.no_grad():
                    test_output = model(test_data)
                    test_loss = loss_function(test_output, test_target)

                    running_test_loss += test_loss.item()
                    test_samples += 1

                    if acc:
                        test_acc += (test_output == test_target).sum().item() / test_target.shape[0]

                model.train()

            running_loss += loss.item()
            samples += 1

            if acc:
                train_acc += (output == target).sum().item() / target.shape[0]

        loss = running_loss / samples
        test_loss = running_test_loss / test_samples

        if acc:
            train_acc /= samples
            test_acc /= test_samples

        print(f"Epoch {epoch}, Train Loss: {loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}")

    print("Training finished")
    model.eval()
    return model


def test_model_against_baseline(model, baseline, test_data, test_target, loss_function, acc=False):
    model.eval()
    baseline.eval()

    model_output = model(test_data)
    baseline_output = baseline(test_data)

    model_loss = loss_function(model_output, test_target)
    baseline_loss = loss_function(baseline_output, test_target)

    print(f"Model loss: {model_loss.item()}, Random Baseline loss: {baseline_loss.item()}")
    if acc:
        model_acc = (model_output == test_target).sum().item() / test_target.shape[0]
        baseline_acc = (baseline_output == test_target).sum().item() / test_target.shape[0]
        print(f"Model accuracy: {model_acc}, Random Baseline accuracy: {baseline_acc}")


def get_mock_db_for_sub_model_1(df, how_many_last_games=10):
    """
    This function serves as a mock database for player history stats
    This would be replaced by a real database with real-time updates of player stats in the real world
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
    print("Sub Model 1 data preprocessing...")

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

    # Split the data into train and test
    train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.2)

    # Normalize the data
    data_mean = train_data.mean(axis=0)
    data_std = train_data.std(axis=0)
    target_mean = train_target.mean(axis=0)
    target_std = train_target.std(axis=0)
    train_data = (train_data - data_mean) / data_std
    train_target = (train_target - target_mean) / target_std
    test_data = (test_data - data_mean) / data_std
    test_target = (test_target - target_mean) / target_std

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

    print("Sub Model 1 data preprocessing finished")

    return train_data, train_target, test_data, test_target, normalization_dict


def sub_model_1(df, force_retrain=False):
    """
    Predict the killing blows, deaths, damage, healing, etc. of a player based on his history
    """
    path_to_model = "models/sub_model_1.pth"
    print("Sub Model 1 started")

    # Transform the data
    train_data, train_target, test_data, test_target, norm_dict = sub_model_1_data_preprocess(df)

    if os.path.exists(path_to_model) and not force_retrain:
        print("Loading pretrained model from cache...")
        # Load the model
        model = SubModel1()
        model.load_state_dict(torch.load(path_to_model, weights_only=True))
    else:
        print("Creating model from scratch...")
        # Train the model
        model = SubModel1()
        loss_function = torch.nn.MSELoss()
        model = train_model(model, train_data, train_target, test_data, test_target, loss_function, lr=0.001, batch_size=128, epochs=10)

        # Test the model against a random baseline
        random_baseline = RandomBaseline1()
        test_model_against_baseline(model, random_baseline, test_data, test_target, loss_function)

        # Save the model
        torch.save(model.state_dict(), path_to_model)

    print("Sub Model 1 finished")

    return model, norm_dict


def sub_model_2_data_preprocess(df, mappings, model_1, norm_dict_1):
    print("Sub Model 2 data preprocessing...")

    # Get the mock database for the player stats
    mock_db = get_mock_db_for_sub_model_1(df, how_many_last_games=10)

    # Get the data and the header
    header = df.columns.values.tolist()
    data = df.to_numpy()

    # For each match, get the players and the map
    match_dict = {}
    for line in data:
        match_id_idx = header.index("match_id")
        player_id_idx = header.index("player_id")
        map_idx = header.index("game_map")
        format_idx = header.index("format")
        winner_idx = header.index("winner")

        match_id = line[match_id_idx]
        won = line[winner_idx]

        if match_id not in match_dict:
            match_dict[match_id] = ([line[map_idx], line[format_idx]], [], [])
        if won:
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
        index = match_dict[key][0][2]

        players_win = np.array(match_dict[key][1], dtype=np.float32)
        players_win = players_win.reshape(-1, players_win.shape[1] * players_win.shape[2])
        players_win = (players_win - norm_dict_1["data_mean"]) / norm_dict_1["data_std"]

        players_loss = np.array(match_dict[key][2], dtype=np.float32)
        players_loss = players_loss.reshape(-1, players_loss.shape[1] * players_loss.shape[2])
        players_loss = (players_loss - norm_dict_1["data_mean"]) / norm_dict_1["data_std"]

        # Randomly shuffle the team order, because the order should not matter for the winning condition
        if np.random.rand() < 0.5:
            players = np.vstack((players_win, players_loss))
            train_target[index].append([1])
        else:
            players = np.vstack((players_loss, players_win))
            train_target[index].append([0])

        output = list(model_1(torch.tensor(players).to(device)).detach().cpu().numpy().reshape(-1))
        output.append(match_dict[key][0][0])

        train_data[index].append(np.array(output))

    # Convert lists to numpy arrays
    for key in train_data:
        try:
            train_data[key] = np.array(train_data[key], dtype=np.float32)
            train_target[key] = np.array(train_target[key], dtype=np.float32)
        except Exception as e:
            print(f"Error in format {key}")
            print(f"len(train_data): {len(train_data[key])}")
            expected_len = len(train_data[key][0])
            for i, item in enumerate(train_data[key]):
                if len(item) != expected_len:
                    print(f"expected_len: {expected_len} vs len(train_data[key][{i}]): {len(item)}")
                    print(f"train_data[key][{i}]: {item}")
            exit(1)

    # Split the data into train and test
    test_data = {}
    test_target = {}
    for key in train_data:
        train_data[key], temp_data, train_target[key], temp_target = train_test_split(train_data[key], train_target[key], test_size=0.2)
        test_data[key] = temp_data
        test_target[key] = temp_target

    # No need for normalization, because the SubModel1 returns normalized players and the map is a number between 0, 10

    # Convert the data to PyTorch tensors
    for key in train_data:
        train_data[key] = torch.tensor(train_data[key], dtype=torch.float32).to(device)
        train_target[key] = torch.tensor(train_target[key], dtype=torch.float32).to(device)
        test_data[key] = torch.tensor(test_data[key], dtype=torch.float32).to(device)
        test_target[key] = torch.tensor(test_target[key], dtype=torch.float32).to(device)

    print("Sub Model 2 data preprocessing finished")

    return train_data, train_target, test_data, test_target


def sub_model_2(df, mappings, model_1, norm_dict_1, force_retrain=False):
    """
    Predict the outcome of a match based on the map and the players in the match
    """
    base_path_to_model = "models/sub_model_2"
    path_to_model = {"10": f"{base_path_to_model}_10.pth", "15": f"{base_path_to_model}_15.pth", "40": f"{base_path_to_model}_40.pth"}
    print("Sub Model 2 started")

    if os.path.exists(path_to_model["10"]) and os.path.exists(path_to_model["15"]) and os.path.exists(path_to_model["40"]) and not force_retrain:
        print("Loading pretrained models from cache...")
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
        print("Creating models from scratch...")
        # Transform the data
        train_data, train_target, test_data, test_target = sub_model_2_data_preprocess(df, mappings, model_1, norm_dict_1)

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
            models[key] = train_model(models[key], train_data[key], train_target[key], test_data[key], test_target[key], loss_function, lr=0.001, batch_size=128, epochs=10, acc=True)

            # Test the model against a random baseline
            test_model_against_baseline(models[key], random_baseline, test_data[key], test_target[key], loss_function, acc=True)

            # Save the model
            torch.save(models[key].state_dict(), path_to_model[key])

    print("Sub Model 2 finished")

    return models
