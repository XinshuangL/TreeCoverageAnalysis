import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_gross_emissions(tree_cover_loss_ha):
    """
    Calculates CO2 emissions based on the known relationship between TreeLoss and CO2 using the calculated 
    prediction equation.
     
    :param tree_cover_loss_ha: float of hectares of tree cover lost
    
    returns:
        float: predicted CO2 emissions
    """
    assert isinstance(tree_cover_loss_ha, float)

    return -59112.48 + 588.21 * tree_cover_loss_ha


def calculate_error_bound(tree_cover_loss_ha):
    """
    Calculates the known error upper bound between TreeLoss and CO2 based on the known
    margin of error, 0.45 * the predicted value.

    :param tree_cover_loss_ha: Float of hactraes of tree cover lost.
    
    returns:
       Float: upper bound in which an error is allowed.
    """
    assert isinstance(tree_cover_loss_ha, float)

    return 0.45 * tree_cover_loss_ha


def validate_predictions(predicted_tree_cover_loss, actual_gross_emissions):
    """
    This function validates the accuracy of the predicted tree cover loss.
     
    :param predicted_tree_cover_loss: Float of predicted hectares of tree cover loss
    :param actual_gross_emissions: Float of the recorded CO2 levels

    returns: 
        Boolean: If the actual CO2 emissions are within the acceptable bounds of the predicted CO2 emissions.
        Float: Upper bound of acceptable error
        Float: Lower bound of acceptable error
        Float: Predicted CO2 emissions based on tree cover loss
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
