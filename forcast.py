# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Print first few rows to confirm the data structure
print(df.head())

# Extract the 'Passengers' column for forecasting
data = df['#Passengers'].values.reshape(-1, 1)

# Normalize the data (feature scaling to the range [0, 1])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create the dataset in the form of X (input) and y (output) for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Define time_step (e.g., using 12 previous months to predict the next month)
time_step = 12

# Prepare the dataset
X, y = create_dataset(scaled_data, time_step)

# Reshape X to be suitable for LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()

# First LSTM layer with Dropout regularization
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the predictions and actual values to the original scale
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate Mean Squared Error (MSE) for model evaluation
mse = mean_squared_error(y_test_rescaled, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted values
plt.plot(y_test_rescaled, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Air Passengers Forecasting with LSTM')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.legend()
plt.show()
