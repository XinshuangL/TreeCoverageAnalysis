############################################################################
# Tree Loss Prediction with Gaussian Process Regression
# Author: Xinshuang Liu
# Email: xil235@ucsd.edu
############################################################################

############################################################################
# This notebook predicts the tree loss of a country in a given year based on the Gaussian process regression model.
# The Gaussian process regression model is a very powerful non-parametric Bayesian model, which takes minimal assumptions for the data.
# Furthermore, it can provide the uncertainty of the prediction, which is useful for future analysis and decision making.
############################################################################

from dataset import TreeCoverLossDataset, DriverTypeDataset
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

import math
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def gp_prediction(x_train, y_train, x_test):
    """
    Perform Gaussian Process Regression (GPR) for prediction.

    :param x_train: Training feature data.
    :param y_train: Training target data.
    :param x_test: Test feature data for prediction.
    :return: Predicted values (mean) and standard deviations.
    """
    # Ensure inputs are 1D arrays reshaped to 2D for GPR
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)

    latest = y_train[-1][0]
    y_train = y_train - latest

    # Define kernel and initialize Gaussian Process Regressor
    # kernel = DotProduct() + WhiteKernel()
    kernel = C(1.0, (1e-2, 1e3)) * (RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + 
            Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)) + \
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)) + \
            WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-5)

    # Fit the model
    gpr.fit(x_train, y_train)

    # Predict with uncertainty
    y_pred, y_std = gpr.predict(x_test, return_std=True)
    y_pred = y_pred + latest
    y_pred = np.abs(y_pred)
    return y_pred, y_std


def eval_prediction(y_pred, y_test):
    """
    Evaluate the prediction using Mean Squared Error (MSE).

    :param y_pred: Predicted values.
    :param y_test: True target values.
    :return: Mean squared error.
    """
    assert len(y_pred) == len(y_test), "Predicted and actual data lengths must match."
    mse = np.mean((y_pred - np.array(y_test)) ** 2)
    return mse


def eval_gp_model(dataset, mode):
    """
    Evaluate the Gaussian Process model with a dataset for either tree loss or CO2 prediction.

    :param dataset: Dataset to evaluate the model.
    :param mode: Mode for evaluation ('tree_loss' or 'co2').
    :return: Root mean squared error (RMSE) of predictions.
    """
    assert mode in ["tree_loss", "co2"], "Mode must be 'tree_loss' or 'co2'."
    target_pos = 1 if mode == "tree_loss" else 2
    mse_list = []

    for train_data, test_data, _ in dataset:
        if len(train_data) == 0 or len(test_data) == 0:
            continue

        # Extract features and targets
        x_train = train_data[:, 0]
        y_train = train_data[:, target_pos]
        x_test = test_data[:, 0]
        y_test = test_data[:, target_pos]

        # Make predictions and calculate MSE
        y_pred, y_std = gp_prediction(x_train, y_train, x_test)
        mse = eval_prediction(y_pred, y_test)
        mse_list.append(mse)

    assert mse_list, "MSE list is empty, ensure valid data is provided."
    return math.sqrt(sum(mse_list) / len(mse_list))


def eval_gp_models():
    """
    Evaluate GP models on multiple datasets and report RMSE for tree loss and CO2 predictions.
    """
    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True
    )
    rmse = eval_gp_model(dataset, "tree_loss")
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, tree loss prediction")

    rmse = eval_gp_model(dataset, "co2")
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, co2 prediction")

    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True
    )
    rmse = eval_gp_model(dataset, "tree_loss")
    print(
        f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, tree loss prediction"
    )

    rmse = eval_gp_model(dataset, "co2")
    print(f"RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, co2 prediction")


def predict_with_gp_models(dataset):
    """
    Predict future values (2020-2030) using GP models for each country in the dataset.

    :param dataset: Dataset to make predictions on.
    :return: Results containing predictions and uncertainties for each country.
    """
    results = {}
    for train_data, test_data, country in dataset:
        try:
            # Tree cover loss predictions
            x_train = train_data[:, 0]
            x_test = test_data[:, 0]
            target_pos = 1
            y_train = train_data[:, target_pos]
            y_test = test_data[:, target_pos]

            # Merge training and test data
            x_merged = torch.cat([x_train, x_test], dim=0)
            y_merged = torch.cat([y_train, y_test], dim=0)
            x_pred = torch.tensor(list(range(2020, 2031)))

            y_pred, y_std = gp_prediction(x_merged, y_merged, x_pred)

            results[country] = {"TreeCoverLoss_ha": {"mean": y_pred, "std": y_std}}

            # CO2 emissions predictions
            target_pos = 2
            y_train = train_data[:, target_pos]
            y_test = test_data[:, target_pos]

            x_merged = torch.cat([x_train, x_test], dim=0)
            y_merged = torch.cat([y_train, y_test], dim=0)
            x_pred = torch.tensor(list(range(2020, 2031)))

            y_pred, y_std = gp_prediction(x_merged, y_merged, x_pred)

            results[country]["GrossEmissions_Co2_all_gases_Mg"] = {
                "mean": y_pred,
                "std": y_std,
            }
        except:
            continue

    return results


def save_results(results, save_dir):
    """
    Save prediction results to a CSV file.

    :param results: Dictionary containing prediction results.
    :param save_dir: Path to save the results file.
    """
    with open(save_dir, "w") as f:
        f.write(
            "CountryCode,Year,TreeCoverLoss_ha,GrossEmissions_Co2_all_gases_Mg,TreeCoverLoss_ha_std,GrossEmissions_Co2_all_gases_Mg_std\n"
        )
        for country, result in results.items():
            for year, loss, co2, loss_std, co2_std in zip(
                range(2020, 2031),
                result["TreeCoverLoss_ha"]["mean"],
                result["GrossEmissions_Co2_all_gases_Mg"]["mean"],
                result["TreeCoverLoss_ha"]["std"],
                result["GrossEmissions_Co2_all_gases_Mg"]["std"],
            ):
                f.write(f"{country},{year},{loss},{co2},{loss_std},{co2_std}\n")


def predict_all_values_using_gp_models():
    """
    Predict all values using GP models for multiple datasets and save the results.
    """
    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True
    )
    results = predict_with_gp_models(dataset)
    save_results(results, "output_data/prediction_ByRegion.csv")

    dataset = TreeCoverLossDataset(
        "input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True
    )
    results = predict_with_gp_models(dataset)
    save_results(results, "output_data/prediction_InPrimaryForest.csv")
