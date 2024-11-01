import os
import numpy as np
import pandas as pd
import torch

"""
SubModel 1) Predict the killing blows, deaths, damage, healing, etc. of a player based on his history
SubModel 2) Predict the outcome of a match based on the map and the players in the match
SubModel 3) Predict the duration of a match based on the map and the players in the match
Model: Combine the three submodels to predict the outcome of a match based on the map and the players in the match
"""


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
        return torch.randn((x.shape[0], 15))


def train_model(model, train_data, train_target, test_data, test_target, loss_function, lr=0.001, batch_size=128, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Training model...")
    for epoch in range(epochs):
        running_test_loss = 0.0
        test_samples = 0
        running_loss = 0.0
        samples = 0

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

                model.train()

            running_loss += loss.item()
            samples += 1

        loss = running_loss / samples
        test_loss = running_test_loss / test_samples
        print(f"Epoch {epoch}, Loss: {loss}, Test Loss: {test_loss}")

    print("Training finished")
    model.eval()
    return model


def test_model_against_baseline(model, baseline, test_data, test_target, loss_function):
    model.eval()
    baseline.eval()

    model_output = model(test_data)
    baseline_output = baseline(test_data)

    model_loss = loss_function(model_output, test_target)
    baseline_loss = loss_function(baseline_output, test_target)

    print(f"Model loss: {model_loss.item()}, Random Baseline loss: {baseline_loss.item()}")


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


def sub_model_1_data_preprocess(df):
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
    split = int(0.8 * train_data.shape[0])
    train_data, test_data = train_data[:split], train_data[split:]
    train_target, test_target = train_target[:split], train_target[split:]

    # Normalize the data
    data_mean = train_data.mean(axis=0)
    data_std = train_data.std(axis=0)
    target_mean = train_target.mean(axis=0)
    target_std = train_target.std(axis=0)
    train_data = (train_data - data_mean) / data_std
    train_target = (train_target - target_mean) / target_std
    test_data = (test_data - data_mean) / data_std
    test_target = (test_target - target_mean) / target_std

    normalization_dict = {
        "data_mean": data_mean,
        "data_std": data_std,
        "target_mean": target_mean,
        "target_std": target_std
    }

    return train_data, train_target, test_data, test_target, normalization_dict


def sub_model_1(df, force_retrain=False):
    """
    Predict the killing blows, deaths, damage, healing, etc. of a player based on his history
    """
    path_to_model = "models/sub_model_1.pth"

    # Transform the data
    train_data, train_target, test_data, test_target, norm_dict = sub_model_1_data_preprocess(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
    train_target = torch.tensor(train_target, dtype=torch.float32).to(device)
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_target = torch.tensor(test_target, dtype=torch.float32).to(device)

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
