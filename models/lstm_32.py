# 1. Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# 2. Load dataset
file_path = os.path.join('data', 'qassim_energy_consumption_cleaned_3833.xlsx')
data = pd.read_excel(file_path)


# 3. Split input and target variables
X = data.iloc[:, :-2].values
y_heating = data.iloc[:, -1].values
y_cooling = data.iloc[:, -2].values


# 4. Normalize data
scaler_X = MinMaxScaler()
scaler_y_heating = MinMaxScaler()
scaler_y_cooling = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_heating_scaled = scaler_y_heating.fit_transform(y_heating.reshape(-1, 1)).flatten()
y_cooling_scaled = scaler_y_cooling.fit_transform(y_cooling.reshape(-1, 1)).flatten()


# 5. Split into training, validation, and test sets (80/10/10)
X_train, X_temp, y_heating_train, y_heating_temp = train_test_split(X_scaled, y_heating_scaled, test_size=0.2, random_state=42)
X_val, X_test, y_heating_val, y_heating_test = train_test_split(X_temp, y_heating_temp, test_size=0.5, random_state=42)

X_train_c, X_temp_c, y_cooling_train, y_cooling_temp = train_test_split(X_scaled, y_cooling_scaled, test_size=0.2, random_state=42)
X_val_c, X_test_c, y_cooling_val, y_cooling_test = train_test_split(X_temp_c, y_cooling_temp, test_size=0.5, random_state=42)


# 6. Reshape input features to 3D format (samples, time steps, features) for LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train_c = X_train_c.reshape((X_train_c.shape[0], 1, X_train_c.shape[1]))
X_val_c = X_val_c.reshape((X_val_c.shape[0], 1, X_val_c.shape[1]))
X_test_c = X_test_c.reshape((X_test_c.shape[0], 1, X_test_c.shape[1]))


# 7. Define the LSTM model architecture
def create_lstm_model():
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    # Layer 1
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 2
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Layer 3
    model.add(LSTM(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(1, activation='linear'))

    # Compile Model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


# 8. Set training configuration and initialize result containers

# List of epoch values to test
epoch_values = [50, 100, 150, 200, 250]
batch_size = 32

# Containers to store results and best models
all_results = []
best_heating_model = None
best_cooling_model = None
best_heating_mse = float('inf')
best_cooling_mse = float('inf')
y_heating_pred_final = None
y_cooling_pred_final = None


# 9. Train and evaluate heating and cooling models across different epoch values
for epochs in epoch_values:
    print(f"\n🔄 Training model with {epochs} epochs...")

    # Train Heating model
    model_heating = create_lstm_model()
    early_stopping_heating = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history_heating = model_heating.fit(
        X_train, y_heating_train,
        validation_data=(X_val, y_heating_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping_heating]
    )
    stopped_epoch_heating = early_stopping_heating.stopped_epoch + 1 if early_stopping_heating.stopped_epoch != 0 else epochs
    y_heating_pred = model_heating.predict(X_test)

    # Evaluate Heating model
    mse_heating = mean_squared_error(y_heating_test, y_heating_pred)
    mae_heating = mean_absolute_error(y_heating_test, y_heating_pred)
    rmse_heating = np.sqrt(mse_heating)
    r2_heating = r2_score(y_heating_test, y_heating_pred)

    # Save best heating model
    if mse_heating < best_heating_mse:
        best_heating_mse = mse_heating
        best_heating_model = model_heating
        y_heating_pred_final = y_heating_pred

    # Train Cooling model
    model_cooling = create_lstm_model()
    early_stopping_cooling = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history_cooling = model_cooling.fit(
        X_train_c, y_cooling_train,
        validation_data=(X_val_c, y_cooling_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping_cooling]
    )
    stopped_epoch_cooling = early_stopping_cooling.stopped_epoch + 1 if early_stopping_cooling.stopped_epoch != 0 else epochs
    y_cooling_pred = model_cooling.predict(X_test_c)

    # Evaluate Cooling model
    mse_cooling = mean_squared_error(y_cooling_test, y_cooling_pred)
    mae_cooling = mean_absolute_error(y_cooling_test, y_cooling_pred)
    rmse_cooling = np.sqrt(mse_cooling)
    r2_cooling = r2_score(y_cooling_test, y_cooling_pred)

    # Save best cooling model
    if mse_cooling < best_cooling_mse:
        best_cooling_mse = mse_cooling
        best_cooling_model = model_cooling
        y_cooling_pred_final = y_cooling_pred

    # store evaluation metrics for this run
    all_results.append({
        'Epochs': epochs,
        'Stopped_Epoch_Heating': stopped_epoch_heating,
        'MSE_Heating': mse_heating,
        'MAE_Heating': mae_heating,
        'RMSE_Heating': rmse_heating,
        'R2_Heating': r2_heating,
        'Stopped_Epoch_Cooling': stopped_epoch_cooling,
        'MSE_Cooling': mse_cooling,
        'MAE_Cooling': mae_cooling,
        'RMSE_Cooling': rmse_cooling,
        'R2_Cooling': r2_cooling
    })


# 10. Save results to Excel files

# Save evaluation metrics across all epoch values
final_results_df = pd.DataFrame(all_results)
final_results_df.to_excel('results/lstm_32_all_results.xlsx', index=False)
print("\nAll evaluation metrics saved to 'results/lstm_32_all_results.xlsx'.")

# Save best predictions for heating and cooling
results_df = pd.DataFrame({
    'Actual Heating Load': y_heating_test,
    'Predicted Heating Load': y_heating_pred_final.flatten(),
    'Actual Cooling Load': y_cooling_test,
    'Predicted Cooling Load': y_cooling_pred_final.flatten()
})
results_df.to_excel('results/lstm_32_predictions.xlsx', index=False)
print("\nPredictions saved to 'results/lstm_32_predictions.xlsx'.")



# 11. Visualize actual vs predicted values for the best model performance

# Identify the best number of epochs based on lowest MSE
best_index_heating = np.argmin([result['MSE_Heating'] for result in all_results])
best_index_cooling = np.argmin([result['MSE_Cooling'] for result in all_results])

best_epochs_heating = all_results[best_index_heating]['Epochs']
best_epochs_cooling = all_results[best_index_cooling]['Epochs']

print(f"Best attempt for Heating Load: Epochs = {best_epochs_heating}")
print(f"Best attempt for Cooling Load: Epochs = {best_epochs_cooling}")

# Set font size for plots
plt.rcParams.update({'font.size': 16})

# Plot: Actual vs Predicted Heating Load
plt.figure(figsize=(12, 6))
plt.scatter(y_heating_test, y_heating_pred_final, alpha=0.5, label='Predicted vs Actual', color='blue')
plt.plot([min(y_heating_test), max(y_heating_test)], [min(y_heating_test), max(y_heating_test)],
         color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Heating Load")
plt.ylabel("Predicted Heating Load")
plt.title(f"Actual vs Predicted Heating Load (Best Epochs: {best_epochs_heating})")
plt.legend()
plt.grid(True)
plt.show()

# Plot: Actual vs Predicted Cooling Load
plt.figure(figsize=(12, 6))
plt.scatter(y_cooling_test, y_cooling_pred_final, alpha=0.5, label='Predicted vs Actual', color='green')
plt.plot([min(y_cooling_test), max(y_cooling_test)], [min(y_cooling_test), max(y_cooling_test)],
         color='red', linestyle='--', label='Ideal Fit')
plt.xlabel("Actual Cooling Load")
plt.ylabel("Predicted Cooling Load")
plt.title(f"Actual vs Predicted Cooling Load (Best Epochs: {best_epochs_cooling})")
plt.legend()
plt.grid(True)
plt.show()

