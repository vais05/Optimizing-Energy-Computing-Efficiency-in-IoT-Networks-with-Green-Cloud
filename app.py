import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = r'C:\Users\Vaishnavi Mohite\Downloads\ML MODEL ON\vmCloud_data.csv'
data = pd.read_csv(file_path)

# Check for missing values and basic information
print("Dataset Information:")
data.info()

# Display the first few rows of the dataset
print("\nPreview of Dataset:")
print(data.head())

# Select only numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[float, int])

# Check correlations for numeric columns only
correlation_matrix = numeric_data.corr()

# Plot heatmap for correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Resource Usage")
plt.show()

# Select relevant features and handle missing values
data_clean = data[['cpu_usage', 'memory_usage', 'energy_efficiency']].dropna()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_clean)

# Convert data into time-series format
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step, 0])  # Predicting CPU usage
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(time_step, X.shape[2])))
model.add(Dense(units=1))  # Output layer

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
print("Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Compute MAE and RMSE
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Dynamic Resource Allocation Logic
def allocate_resources(predicted_usage, current_usage, threshold=0.8):
    if predicted_usage > threshold * current_usage:
        return 'Scale Up'
    elif predicted_usage < 0.5 * current_usage:
        return 'Scale Down'
    else:
        return 'Maintain'

# Predict for the entire dataset (not just 5 test cases)
all_predictions = model.predict(X)

# Reverse scale predictions for interpretation
scaled_predictions = scaler.inverse_transform(
    np.column_stack((all_predictions, np.zeros_like(all_predictions), np.zeros_like(all_predictions)))
)

# Simulate decisions for the entire dataset
allocations = [allocate_resources(pred, curr) for pred, curr in zip(scaled_predictions[:, 0], y)]

# Save the predictions to CSV
predictions_df = pd.DataFrame(scaled_predictions, columns=['predicted_cpu', 'dummy_1', 'dummy_2'])
output_file_path = r'C:\Users\Vaishnavi Mohite\Downloads\ML_MODEL_ON\predictions.csv'
predictions_df.to_csv(output_file_path, index=False)

# Print dynamic resource allocation decisions for the entire dataset
print("\nDynamic Resource Allocation Decisions for Entire Dataset:")
for i, alloc in enumerate(allocations):
    print(f"Test Case {i + 1}: {alloc}")

# Plot training vs validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot predicted vs actual CPU usage
plt.figure(figsize=(12, 6))
plt.plot(scaled_predictions[:, 0], label='Predicted CPU Usage', color='blue')
plt.plot(y, label='Actual CPU Usage', color='red', linestyle='dashed')
plt.title("Predicted vs Actual CPU Usage for Entire Dataset")
plt.xlabel('Time')
plt.ylabel('CPU Usage')
plt.legend()
plt.show()

# Plot energy efficiency over time
plt.figure(figsize=(12, 6))
plt.plot(data['energy_efficiency'], label='Energy Efficiency', color='green')
plt.title("Energy Efficiency Over Time")
plt.xlabel('Time')
plt.ylabel('Energy Efficiency')
plt.legend()
plt.show()

# Confirm the CSV file creation
print(f"Predictions saved to: {output_file_path}")
