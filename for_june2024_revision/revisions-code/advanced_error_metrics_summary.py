import pickle
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
pd.set_option("display.precision", 2)


# print current working directory
print(os.getcwd())
# set current working directory to "C:/automatic-rainfall-runoff-models/for_june2024_revision/revisions-code"
os.chdir("C:/automatic-rainfall-runoff-models/for_june2024_revision/revisions-code")
print(os.getcwd())

usgs_error_metrics = pd.read_csv("usgs_modpods_results/ensemble.csv",index_col=0)
camels_error_metrics = pd.read_csv("camels_modpods_results/ensemble.csv",index_col=0)

print(usgs_error_metrics)
print(camels_error_metrics)

# drop the "model" column
usgs_error_metrics = usgs_error_metrics.drop(columns="model")
# dimensioned error metrics are in feet (stage), convert to meters
# dimensioned error metrics are just MAE and RMSE
usgs_error_metrics["MAE"] = usgs_error_metrics["MAE"] * 0.3048
usgs_error_metrics["RMSE"] = usgs_error_metrics["RMSE"] * 0.3048

camels_error_metrics = camels_error_metrics.drop(columns="model")

pd.set_option('display.max_columns', None)
# set pandas print option to 2 decimal places, scientific notation
pd.set_option('display.float_format', '{:.2e}'.format)

pd.set_option('display.float_format', '{:.4f}'.format)
# print the summary statistics for each column
# drop HFV10 from usgs
usgs_error_metrics = usgs_error_metrics.drop(columns="HFV10")
print("USGS Error Metrics Summary")
print(usgs_error_metrics.describe())


print("CAMELS Error Metrics Summary")
print(camels_error_metrics.describe())
# dimensioned error metrics (on OBS_RUN) are mm / day - per https://hess.copernicus.org/articles/19/209/2015/hess-19-209-2015.pdf

# just print the median values for CAMELS
pd.set_option('display.float_format', '{:.2f}'.format)

print("CAMELS Median Values")
print(camels_error_metrics.median())