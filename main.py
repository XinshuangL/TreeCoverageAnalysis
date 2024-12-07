############################################################################
# This is the main file for running the modular .py files.
# Author: Xinshuang Liu, Haoyu Hu, Emma, Wrightson, Cao, Qi
# Email: xil235@ucsd.edu, hah034@ucsd.edu, ewrightson@ucsd.edu, q9cao@ucsd.edu
############################################################################

############################################################################
# Import the modules
############################################################################
from prediction_baselines_statistical_methods import (
    test_baseline1,
    test_baseline2,
    test_baseline3,
    test_baseline4,
)
from Gaussian_process_regression import (
    eval_gp_models,
    predict_all_values_using_gp_models,
)
import correlation_analysis

############################################################################
# Section 1: Test the models
############################################################################

# Section 1.1: Test the baselines
print("-" * 10)
print("Section 1.1: Test the baselines")
print("-" * 10)

print("test baseline 1")
test_baseline1()
print("-" * 10)

print("test baseline 2")
test_baseline2()
print("-" * 10)

print("test baseline 3")
test_baseline3()
print("-" * 10)

print("test baseline 4")
test_baseline4()
print("-" * 10)

# Section 1.2: Evaluate gp models
print("Evaluate gp models")
eval_gp_models()
print("-" * 10)

############################################################################
# Section 2: Predict the tree loss and co2 emissions and then save
# their values in the output_data file, prediction_ByRegion.csv,
# and prediction_InPrimaryForest.csv 
############################################################################

predict_all_values_using_gp_models()

############################################################################
# Section 3: Correlation analysis between the tree loss and co2 emissions
############################################################################

# Section 3.1: Correlation coefficient computing
print("Correlation coefficient computing")
correlation_analysis.correlation_coefficient_computing()

# Section 3.2: Perform the OLS regression analysis
print("Perform the OLS regression analysis")
correlation_analysis.OLS_function_for_correlation_analysis()
