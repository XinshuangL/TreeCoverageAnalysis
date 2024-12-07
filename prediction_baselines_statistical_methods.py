############################################################################
# Four statistical baseline methods for tree loss and co2 prediction.
# Author: Xinshuang Liu
# Email: xil235@ucsd.edu
############################################################################

############################################################################
# Prediction Baseline Methods
# This notebook contains four statistical baseline methods for tree loss and co2 prediction.

# For statistical methods, there are four baseline methods to predict the values:
# - Baseline 1: Global mean
# - Baseline 2: Local mean
# - Baseline 3: Latest value
# - Baseline 4: Mixture of local mean and latest value

# The reason of using "mixture of local mean and latest value" is because I find the global mean performs bad.
############################################################################

from dataset import TreeCoverLossDataset, DriverTypeDataset
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")


def eval_prediction(y_pred, y_test):
    """
    Compute MSE between predictions and gt

    :param y_pred: predicted values
    :param y_test: gt values
    :return: MSE score
    """
    assert len(y_pred) == len(y_test)

    mse = np.mean((y_pred - np.array(y_test)) ** 2)
    return mse


def eval_gp_model(dataset, mode, model, params):
    """
    Evaluate the GPR model

    :param dataset: dataset
    :param mode: "tree_loss" or "co2" 
    :param model: model
    :param params: some other settings
    :return: RMSE
    """
    assert mode in ["tree_loss", "co2"]

    target_pos = 1 if mode == "tree_loss" else 2
    mse_list = []
    for train_data, test_data, _ in dataset:
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        x_train = train_data[:, 0]
        y_train = train_data[:, target_pos]
        x_test = test_data[:, 0]
        y_test = test_data[:, target_pos]
        y_pred = model(x_train, y_train, x_test, params)
        mse = eval_prediction(y_pred, y_test)
        mse_list.append(mse)
    return math.sqrt(sum(mse_list) / len(mse_list))


############################################################################
# Baseline Method 1: Global Mean
############################################################################
def baseline1(x_train, y_train, x_test, params):
    """
    Baseline method #1

    :param x_train: trian input
    :param y_train: train target
    :param x_test: gt
    :param params: some other settings
    :return: predictions
    """
    assert len(x_train) == len(y_train)

    global_mean = params["global_mean"]
    y_pred = np.zeros_like(x_test) + global_mean
    return y_pred


def get_target_mean(dataset, pos):
    """
    Gets the mean values

    :param dataset: dataset
    :param pos: position
    :return: mean
    """
    assert pos >= 0

    sum_value = 0
    count_value = 0
    for train_data, test_data, _ in dataset:
        try:
            sum_value += float(train_data[:, pos].sum())
            count_value += train_data[:, pos].shape[0]
        except:
            pass
    return sum_value / count_value


def test_baseline1():
    """
    Test the baseline method #1
    """
    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True
    )

    global_mean = get_target_mean(dataset, 1)
    rmse = eval_gp_model(dataset, "tree_loss", baseline1, {"global_mean": global_mean})
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, tree loss prediction")
    global_mean = get_target_mean(dataset, 2)
    rmse = eval_gp_model(dataset, "co2", baseline1, {"global_mean": global_mean})
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, co2 prediction")

    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True
    )

    global_mean = get_target_mean(dataset, 1)
    rmse = eval_gp_model(dataset, "tree_loss", baseline1, {"global_mean": global_mean})
    print(
        f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, tree loss prediction"
    )
    global_mean = get_target_mean(dataset, 2)
    rmse = eval_gp_model(dataset, "co2", baseline1, {"global_mean": global_mean})
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, co2 prediction")


############################################################################
# Baseline Method 2: Local Mean
############################################################################
def baseline2(x_train, y_train, x_test, params):
    """
    Baseline method #2

    :param x_train: trian input
    :param y_train: train target
    :param x_test: gt
    :param params: some other settings
    :return: predictions
    """
    assert len(x_train) == len(y_train)

    local_mean = y_train.mean()
    y_pred = np.zeros_like(x_test) + float(local_mean)
    return y_pred


def test_baseline2():
    """
    Test the baseline method #2
    """
    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True
    )
    rmse = eval_gp_model(dataset, "tree_loss", baseline2, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, tree loss prediction")
    rmse = eval_gp_model(dataset, "co2", baseline2, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, co2 prediction")

    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True
    )
    rmse = eval_gp_model(dataset, "tree_loss", baseline2, None)
    print(
        f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, tree loss prediction"
    )
    rmse = eval_gp_model(dataset, "co2", baseline2, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, co2 prediction")


############################################################################
# Baseline Method 3: Latest
############################################################################
def baseline3(x_train, y_train, x_test, params):
    """
    Baseline method #3

    :param x_train: trian input
    :param y_train: train target
    :param x_test: gt
    :param params: some other settings
    :return: predictions
    """
    assert len(x_train) == len(y_train)

    latest = y_train.view(-1)[-1]
    y_pred = np.zeros_like(x_test) + float(latest)
    return y_pred


def test_baseline3():
    """
    Test the baseline method #3
    """
    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True
    )

    rmse = eval_gp_model(dataset, "tree_loss", baseline3, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, tree loss prediction")
    rmse = eval_gp_model(dataset, "co2", baseline3, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, co2 prediction")

    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True
    )

    rmse = eval_gp_model(dataset, "tree_loss", baseline3, None)
    print(
        f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, tree loss prediction"
    )
    rmse = eval_gp_model(dataset, "co2", baseline3, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, co2 prediction")


############################################################################
# Baseline Method 4: Mixture of Local Mean and Latest Value
############################################################################
def baseline4(x_train, y_train, x_test, params):
    """
    Baseline method #4

    :param x_train: trian input
    :param y_train: train target
    :param x_test: gt
    :param params: some other settings
    :return: predictions
    """
    assert len(x_train) == len(y_train)

    return (
        baseline2(x_train, y_train, x_test, params)
        + baseline3(x_train, y_train, x_test, params)
    ) / 2


def test_baseline4():
    """
    Test the baseline method #4
    """
    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True
    )

    rmse = eval_gp_model(dataset, "tree_loss", baseline4, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, tree loss prediction")
    rmse = eval_gp_model(dataset, "co2", baseline4, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, co2 prediction")

    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True
    )

    rmse = eval_gp_model(dataset, "tree_loss", baseline4, None)
    print(
        f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, tree loss prediction"
    )
    rmse = eval_gp_model(dataset, "co2", baseline4, None)
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, co2 prediction")
