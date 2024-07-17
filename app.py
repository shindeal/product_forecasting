# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:15:27 2024

@author: anjali.shinde
"""

import streamlit as st
import pandas as pd
import pyodbc
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima import auto_arima

st.title('Product Sales Forecasting')

# Load your data from SQL
@st.cache
def load_data():
    connection_string = r"Driver={SQL Server};Server=L1SQLS1601P\SpeedyDWAnalytic;Database=Speedy_Models;Trusted_Connection=yes;"
    conn = pyodbc.connect(connection_string)
    query = """
    SELECT  [Status], [Product key], [Start Date],
        DATEPART(YEAR, [Start Date Time]) AS Year,
        DATEPART(MONTH, [Start Date Time]) AS Month,
        COUNT(*) AS Count
    FROM 
        [L1SQLS1601p\SpeedyDWAnalytic].[Speedy_Models].[dbo].[FACT_DEVICE_LOCATION_STATUSES] WITH (NOLOCK)
    WHERE 
        [Status] IN ('SOLD') 
        AND DATEPART(YEAR, [Start Date Time]) IN ('2022','2023','2024') 
        AND [Product key] IN ('01/0072', '01/0073')
    GROUP BY  [Status], [Product key], [Start Date],
        DATEPART(YEAR, [Start Date Time]), 
        DATEPART(MONTH, [Start Date Time])
    ORDER BY [Product key], [Start Date],
        DATEPART(YEAR, [Start Date Time]), 
        DATEPART(MONTH, [Start Date Time]);
    """
    data = pd.read_sql_query(query, conn)
    conn.close()
    data['Start Date'] = pd.to_datetime(data['Start Date'], format='%Y-%m-%d')
    data.set_index('Start Date', inplace=True)
    return data

data = load_data()

# Define the date ranges for train and test sets
train_end_date = pd.to_datetime('2024-01-31')
test_start_date = pd.to_datetime('2024-02-01')
test_end_date = pd.to_datetime('2024-07-31')

# Select product key
product_key = st.selectbox("Select Product Key", data['Product key'].unique())

product_data = data[data['Product key'] == product_key]

# Split the data into train and test sets
train = product_data[product_data.index <= train_end_date]
test = product_data[(product_data.index >= test_start_date) & (product_data.index <= test_end_date)]

y_train = train['Count']
y_test = test['Count']

# Fit the auto-ARIMA model
auto_arima_model = auto_arima(y_train, seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
st.write(f"Auto-ARIMA summary for {product_key}:\n", auto_arima_model.summary())

# Generate predictions for the test data
y_pred_arima = auto_arima_model.predict(n_periods=len(test))
test['Predicted_ARIMA'] = y_pred_arima

# Calculate RMSE and MAE for auto-ARIMA
rmse_arima = np.sqrt(mean_squared_error(y_test, y_pred_arima))
mae_arima = mean_absolute_error(y_test, y_pred_arima)

# Fit the SARIMAX model
sarimax_model = SARIMAX(y_train, order=(0, 0, 1), seasonal_order=(0, 0, 1, 12))
sarimax_model_fit = sarimax_model.fit(disp=False)

# Generate predictions for the test data
y_pred_sarimax = sarimax_model_fit.get_forecast(len(test)).predicted_mean
test['Predicted_SARIMAX'] = y_pred_sarimax

# Calculate RMSE and MAE for SARIMAX
rmse_sarimax = np.sqrt(mean_squared_error(y_test, y_pred_sarimax))
mae_sarimax = mean_absolute_error(y_test, y_pred_sarimax)

# Display results
st.write(f"Product: {product_key}")
st.write(f"RMSE for auto-ARIMA: {rmse_arima}")
st.write(f"MAE for auto-ARIMA: {mae_arima}")
st.write(f"RMSE for SARIMAX: {rmse_sarimax}")
st.write(f"MAE for SARIMAX: {mae_sarimax}")

# Forecast for the next 6 months using auto-ARIMA
forecast_steps = 6
forecast_arima = auto_arima_model.predict(n_periods=forecast_steps)
forecast_index = pd.date_range(start=test_end_date + pd.DateOffset(days=1), periods=forecast_steps, freq='MS')
forecast_df_arima = pd.DataFrame(forecast_arima.values, index=forecast_index, columns=['Forecast_ARIMA'])

# Forecast for the next 6 months using SARIMAX
forecast_sarimax = sarimax_model_fit.forecast(steps=forecast_steps)
forecast_df_sarimax = pd.DataFrame(forecast_sarimax.values, index=forecast_index, columns=['Forecast_SARIMAX'])

# Plot the actual vs predicted values
plt.figure(figsize=(14, 7))
plt.plot(train.index, y_train, label='Training Data')
plt.plot(test.index, y_test, label='Actual Test Data')
plt.plot(test.index, y_pred_arima, label='Predicted ARIMA', color='red')
plt.plot(test.index, y_pred_sarimax, label='Predicted SARIMAX', color='green')
plt.legend()
plt.title(f"Product: {product_key} - Actual vs Predicted")
plt.xlabel('Date')
plt.ylabel('Count')

# Display the plot in Streamlit
st.pyplot(plt)

# Display forecasted values
st.write(f"Forecasted values for {product_key} using auto-ARIMA:\n", forecast_df_arima)
st.write(f"Forecasted values for {product_key} using SARIMAX:\n", forecast_df_sarimax)



