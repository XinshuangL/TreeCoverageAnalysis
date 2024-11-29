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
import math
import torch
import warnings

warnings.filterwarnings("ignore")

def gp_prediction(x_train, y_train, x_test):
    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    x_train = np.array(x_train).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    x_test = np.array(x_test).reshape(-1, 1)
    gpr.fit(x_train, y_train)
    y_pred, y_std = gpr.predict(x_test, return_std=True)
    return y_pred, y_std

def eval_prediction(y_pred, y_test):
    mse = np.mean((y_pred - np.array(y_test)) ** 2)
    return mse

def eval_gp_model(dataset, mode):
    target_pos = 1 if mode == "tree_loss" else 2
    mse_list = []
    for train_data, test_data, _ in dataset:
        if len(train_data) == 0 or len(test_data) == 0:
            continue
        x_train = train_data[:, 0]
        y_train = train_data[:, target_pos]
        x_test = test_data[:, 0]
        y_test = test_data[:, target_pos]
        y_pred, y_std = gp_prediction(x_train, y_train, x_test)
        mse = eval_prediction(y_pred, y_test)
        mse_list.append(mse)
    return math.sqrt(sum(mse_list) / len(mse_list))

def eval_gp_models():
    dataset = TreeCoverLossDataset("input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True)
    rmse = eval_gp_model(dataset, "tree_loss")
    print(f'RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, tree loss prediction')
    rmse = eval_gp_model(dataset, "co2")
    print(f'RMSE: {rmse}, TreeCoverLoss_2001-2020_ByRegion, co2 prediction')

    dataset = TreeCoverLossDataset("input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True)
    rmse = eval_gp_model(dataset, "tree_loss")
    print(f'RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, tree loss prediction')
    rmse = eval_gp_model(dataset, "co2")
    print(f'RMSE: {rmse}, TreeCoverLoss_2001-2020 _InPrimaryForest, co2 prediction')

def predict_with_gp_models(dataset):
    results = {}
    for train_data, test_data, country in dataset:
        try:
            x_train = train_data[:, 0]
            x_test = test_data[:, 0]

            target_pos = 1
            y_train = train_data[:, target_pos]
            y_test = test_data[:, target_pos]

            x_merged = torch.cat([x_train, x_test], dim=0)
            y_merged = torch.cat([y_train, y_test], dim=0)
            x_pred = torch.tensor(list(range(2020, 2031)))

            y_pred, y_std = gp_prediction(x_merged, y_merged, x_pred)

            results[country] = {
                'TreeCoverLoss_ha': {
                    'mean': y_pred,
                    'std': y_std
                }
            }

            target_pos = 2
            y_train = train_data[:, target_pos]
            y_test = test_data[:, target_pos]

            x_merged = torch.cat([x_train, x_test], dim=0)
            y_merged = torch.cat([y_train, y_test], dim=0)
            x_pred = torch.tensor(list(range(2020, 2031)))

            y_pred, y_std = gp_prediction(x_merged, y_merged, x_pred)

            results[country]['GrossEmissions_Co2_all_gases_Mg'] = {
                'mean': y_pred,
                'std': y_std
            }
        except:
            pass

    return results

def save_results(results, save_dir):
    with open(save_dir, 'w') as f:
        f.write('CountryCode,Year,TreeCoverLoss_ha,GrossEmissions_Co2_all_gases_Mg,TreeCoverLoss_ha_std,GrossEmissions_Co2_all_gases_Mg_std\n')
        for country, result in results.items():
            for year, loss, co2, loss_std, co2_std in zip(range(2020, 2031), result['TreeCoverLoss_ha']['mean'], result['GrossEmissions_Co2_all_gases_Mg']['mean'], result['TreeCoverLoss_ha']['std'], result['GrossEmissions_Co2_all_gases_Mg']['std']):
                f.write(f'{country},{year},{loss},{co2},{loss_std},{co2_std}\n')

def predict_all_values_using_gp_models():
    dataset = TreeCoverLossDataset("input_data/TreeCoverLoss_2001-2020_ByRegion.csv", split_train_test=True)
    results = predict_with_gp_models(dataset)
    save_results(results, 'output_data/prediction_ByRegion.csv')
    dataset = TreeCoverLossDataset("input_data/TreeCoverLoss_2001-2020 _InPrimaryForest.csv", split_train_test=True)
    results = predict_with_gp_models(dataset)
    save_results(results, 'output_data/prediction_InPrimaryForest.csv')
