import pandas as pd
import numpy as np
import os

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge

class CpuForecaster:

    def __init__(self):
        pass

    def forecast(self, vm_cpu_usage):
        cpu_predictions = dict()

        for vm_id, historic in vm_cpu_usage.items():
            vm_historic = np.array(historic)
            
            # Convert dataframe to Time Series array. Fill missing dates with NaN value
            df = pd.DataFrame(vm_historic, columns=["time", "cpu"])

            # Convert dataframe to Time Series array. Fill missing dates with NaN value
            cpu_series = TimeSeries.from_dataframe(df, "time", "cpu", fill_missing_dates=True)

            # Fill NaN values in time series using interpolation
            cpu_series = fill_missing_values(cpu_series)

            # Define the forecasting model --> Bayesian Ridge Regression
            model = RegressionModel(lags=24,  model=BayesianRidge())

            # Fit the model 
            model.fit(cpu_series)

            # Call model predict function --> predict next 24 hours
            cpu_forecast = model.predict(n=24)

            # Convert predicted series to dataframe
            df = cpu_forecast.pd_dataframe()

            # Add results to dict
            cpu_predictions[vm_id] = df[df.columns[0]].values.tolist()

        return cpu_predictions  
