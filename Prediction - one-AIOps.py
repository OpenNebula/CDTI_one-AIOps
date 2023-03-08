#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries

import pandas as pd
import os

from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.models import RegressionModel
from sklearn.linear_model import BayesianRidge


# In[ ]:


# WORKING directory 
work_dir = "/Users/rafa/Documents/Investigacion/VM-Traces/aux/"

# INPUT dir: Directory with CPU usage (%) traces
# One csv file for every VM (vm_id.csv)
input_dir = work_dir + "trace_files/"

    # input csv files included in input directory (one file per VM)
    # They include hourly traces for CPU usage (%) for several days (e.g. 30 days)
    # Format:
    # time,cpu
    # 2017-01-01 00:00:00,0.1150136
    # 2017-01-01 01:00:00,0.1146467
    # ...
    # 2017-01-30 22:00:00,0.118682
    # 2017-01-30 23:00:00,0.1166679

# OUTPUT dir: Directory with CPU usage (%) predcitions
# One csv file for every VM (vm_id.csv)
output_dir = work_dir + "predictions/"

    # output csv files included in output directory (one file per VM)
    # They include hourly forecast of CPU ussage (%) for next 24 hours
    # Format: 
    # time,cpu
    # 2017-01-31 00:00:00,0.1289253844258983
    # 2017-01-31 01:00:00,0.13198002226021172
    # ...
    # 2017-01-31 23:00:00,0.1299979908043143



# Check if output_dir exists
if os.path.exists(output_dir):
    # if output_dir exists, clean it
    for file in os.listdir(output_dir):
        os.remove(output_dir + file) 
    # if output_dir does not exist, create it
else:
    os.mkdir(output_dir)


# In[ ]:


# Read .csv trace files and convert to time series 
cpu_train_series_list = []
mem_train_series_list = []

cpu_val_series_list = []
mem_val_series_list = []

for file in os.listdir(input_dir):
    # read files from input dir
    if file.endswith('.csv'):
        # vm_id = file name without extension .csv
        vm_id = os.path.splitext(file)[0]
        
        print("CPU forecasting for ", vm_id)
        file_path = input_dir + file
        
        # Read csv trace file and convert it to dataframe
        df = pd.read_csv(file_path, delimiter=",", engine='python')
        
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
        
        # Save dataframe to output csv file
        file_path = output_dir + file
        df.to_csv(file_path, index=True)    
    

