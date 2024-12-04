import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_gross_emissions(tree_cover_loss_ha):
    """
    the known relationship between TreeLoss and CO2
    """
    assert isinstance(tree_cover_loss_ha, float)

    return -59112.48 + 588.21 * tree_cover_loss_ha


def calculate_error_bound(tree_cover_loss_ha):
    """
    the known error upper bound between TreeLoss and CO2
    """
    assert isinstance(tree_cover_loss_ha, float)

    return 0.45 * tree_cover_loss_ha


def validate_predictions(predicted_tree_cover_loss, actual_gross_emissions):
    """
    validate the accuracy of the predicted tree cover loss
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
