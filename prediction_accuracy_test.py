import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_gross_emissions(tree_cover_loss_ha):
    """
    Calculates the known relationship between TreeLoss and CO2, takes in a 
    loss of tree loss as a float, 'tree_cover_loss_ha' and returns
    the calculated CO2 emissions based on the calculated correlation
    equation.
    """
    assert isinstance(tree_cover_loss_ha, float)

    return -59112.48 + 588.21 * tree_cover_loss_ha


def calculate_error_bound(tree_cover_loss_ha):
    """
    Calculates the known error upper bound between TreeLoss and CO2, 
    takes in a float 'tree_cover_loss_ha' and calculates the error
    bounds by multiplying by 0.45, the calculated error bound parameter, and returns that value.
    """
    assert isinstance(tree_cover_loss_ha, float)

    return 0.45 * tree_cover_loss_ha


def validate_predictions(predicted_tree_cover_loss, actual_gross_emissions):
    """
    This function validates the accuracy of the predicted tree cover loss. It takes in 
    'predicted_tree_cover_loss1, a float, and 'actual_gross_emissions', another float 
    and then calculates the upper and lower bounds and predicts the CO2 emissions
    based off of the tree cover loss, and then returns a boolean if the
    actual value is within the error bounds of the predicted value.
    """
    assert isinstance(predicted_tree_cover_loss, float) and isinstance(
        actual_gross_emissions, float
    )

    predicted_gross_emissions = calculate_gross_emissions(predicted_tree_cover_loss)
    error_bound = calculate_error_bound(predicted_tree_cover_loss)

    lower_bound = predicted_gross_emissions - error_bound
    upper_bound = predicted_gross_emissions + error_bound

    is_accurate = (actual_gross_emissions >= lower_bound) & (
        actual_gross_emissions <= upper_bound
    )

    return is_accurate, lower_bound, upper_bound, predicted_gross_emissions
