# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
### Date: 19-05-2026

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('/content/Walmart_Sales.csv')
data=data.head(200)
# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%Y")

# Plot Weekly Sales Time Series
plt.plot(data['Date'], data['Weekly_Sales'])
plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('Weekly Sales Time Series')
plt.show()

# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries)

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')

    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

# Check stationarity
check_stationarity(data['Weekly_Sales'])

# ACF Plot
plot_acf(data['Weekly_Sales'])
plt.show()

# PACF Plot
plot_pacf(data['Weekly_Sales'])
plt.show()

# Split data into train and test
train_size = int(len(data) * 0.8)

train = data['Weekly_Sales'][:train_size]
test = data['Weekly_Sales'][train_size:]

# Build SARIMA Model
sarima_model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

sarima_result = sarima_model.fit()

# Predictions
predictions = sarima_result.predict(
    start=len(train),
    end=len(train) + len(test) - 1
)

# RMSE Calculation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)

print('RMSE:', rmse)

# Plot Actual vs Predicted
plt.plot(data['Date'][train_size:], test, label='Actual')
plt.plot(data['Date'][train_size:], predictions, color='red', label='Predicted')

plt.xlabel('Date')
plt.ylabel('Weekly Sales')
plt.title('SARIMA Model Predictions')

plt.legend()
plt.show()
```
### OUTPUT:
<img width="1025" height="713" alt="image" src="https://github.com/user-attachments/assets/8eabe311-d4b1-4362-9737-20e732b42672" />

<img width="1012" height="540" alt="image" src="https://github.com/user-attachments/assets/e7cf4315-1011-4eee-8fb2-611cdefba606" />
<img width="1078" height="538" alt="image" src="https://github.com/user-attachments/assets/2f869007-8825-4df0-937d-08721491ae33" />
<img width="1036" height="587" alt="image" src="https://github.com/user-attachments/assets/3bdcd626-4d42-4436-9b31-f82347a356d5" />

### RESULT:
Thus the program run successfully based on the SARIMA model.
