import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
from autots import AutoTS

# not sure if i'll end up using this. but what's written below seems to "work"

# example record, would actually be looped over all records
filepath = "G:/My Drive/PhD Admin and Notes/paper1/CAMELS/basin_timeseries_v1p2_modelOutput_daymet/model_output_daymet/model_output/flow_timeseries/daymet/01/01013500_05_model_output.txt"
df = pd.read_csv(filepath, sep='\s+')
df.rename({'YR':'year','MNTH':'month','DY':'day','HR':'hour'},axis=1,inplace=True)
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
df.set_index('datetime',inplace=True)
df.set_index(pd.DatetimeIndex(df.index), inplace=True)
print(df)
df = df[['RAIM','OBS_RUN']]

model = AutoTS(
    forecast_length=1,
    frequency='infer',
    max_generations=4,
    num_validations=2,
    validation_method="backwards"
)
model = model.fit(
    df
)

prediction = model.predict()
# plot a sample
prediction.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0],
                start_date="1990-01-01")
# Print the details of the best model
print(model)

# point forecasts dataframe
forecasts_df = prediction.forecast
# upper and lower forecasts
forecasts_up, forecasts_low = prediction.upper_forecast, prediction.lower_forecast

# accuracy of all tried model results
model_results = model.results()
# and aggregated from cross validation
validation_results = model.results("validation")