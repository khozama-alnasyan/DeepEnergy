# 1. Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# 2. Load dataset from the data folder
file_path = os.path.join('data', 'qassim_energy_consumption_cleaned_3833.xlsx')
data = pd.read_excel(file_path)


# 3. Separate input features (X) and target outputs (y)
X = data.iloc[:, :-2].values             # All columns except last two
y_heating = data.iloc[:, -1].values      # Last column (Heating Load)
y_cooling = data.iloc[:, -2].values      # Second to last column (Cooling Load)


# 4. Normalize features and targets using MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y_heating = MinMaxScaler()
scaler_y_cooling = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_heating_scaled = scaler_y_heating.fit_transform(y_heating.reshape(-1, 1)).flatten()
y_cooling_scaled = scaler_y_cooling.fit_transform(y_cooling.reshape(-1, 1)).flatten()

# Reshape input to 3D format for Conv1D: (samples, timesteps, features)
X_scaled_cnn = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))


# 5. Split the dataset into training, validation, and test sets (80/10/10)

# For heating
X_train, X_temp, y_heating_train, y_heating_temp = train_test_split(X_scaled_cnn, y_heating_scaled, test_size=0.2, random_state=42)
X_val, X_test, y_heating_val, y_heating_test = train_test_split(X_temp, y_heating_temp, test_size=0.5, random_state=42)

# For cooling
X_train_c, X_temp_c, y_cooling_train, y_cooling_temp = train_test_split(X_scaled_cnn, y_cooling_scaled, test_size=0.2, random_state=42)
X_val_c, X_test_c, y_cooling_val, y_cooling_test = train_test_split(X_temp_c, y_cooling_temp, test_size=0.5, random_state=42)


# 6. Define the CNN model architecture
def create_cnn_model():
    model = Sequential()
    model.add(Input(shape=(X_scaled_cnn.shape[1], 1)))

    # Conv1D layer 1
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Conv1D layer 2
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Conv1D layer 3
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Flatten layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(1, activation='linear'))

    # Compile Model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


# 7. Set training configuration and initialize result containers

# List of epoch values to test
epoch_values = [50, 100, 150, 200, 250]
batch_size = 64

# Containers to store results and best models
results = []
best_heating_mse = float('inf')
best_cooling_mse = float('inf')
y_heating_pred_final = None
y_cooling_pred_final = None


# 8. Train and evaluate heating and cooling models across different epoch values

for epochs in epoch_values:
    print(f"\nTraining with {epochs} epochs...")

    # Train heating model
    model_heating = create_cnn_model()
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

    # Train cooling model
    model_cooling = create_cnn_model()
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

    # Evaluate heating model
    mse_heating = mean_squared_error(y_heating_test, y_heating_pred)
    mae_heating = mean_absolute_error(y_heating_test, y_heating_pred)
    rmse_heating = np.sqrt(mse_heating)
    r2_heating = r2_score(y_heating_test, y_heating_pred)

    # Evaluate cooling model
    mse_cooling = mean_squared_error(y_cooling_test, y_cooling_pred)
    mae_cooling = mean_absolute_error(y_cooling_test, y_cooling_pred)
    rmse_cooling = np.sqrt(mse_cooling)
    r2_cooling = r2_score(y_cooling_test, y_cooling_pred)

    # Save best predictions based on MSE
    if mse_heating < best_heating_mse:
        best_heating_mse = mse_heating
        y_heating_pred_final = y_heating_pred

    if mse_cooling < best_cooling_mse:
        best_cooling_mse = mse_cooling
        y_cooling_pred_final = y_cooling_pred

    # Store evaluation metrics for this run
    results.append({
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


# 9. Save training results and predictions to Excel files

# Save evaluation metrics across all epoch values
final_results_df = pd.DataFrame(results)
final_results_df.to_excel('results/cnn_64_all_results.xlsx', index=False)

# Save best predictions for heating and cooling
results_df = pd.DataFrame({
    'Actual Heating Load': y_heating_test,
    'Predicted Heating Load': y_heating_pred_final.flatten(),
    'Actual Cooling Load': y_cooling_test,
    'Predicted Cooling Load': y_cooling_pred_final.flatten()
})
results_df.to_excel('results/cnn_64_predictions.xlsx', index=False)


# 10. Visualize actual vs predicted values for the best model performance

# Identify best epochs from stored results
best_index_heating = np.argmin([result['MSE_Heating'] for result in results])
best_index_cooling = np.argmin([result['MSE_Cooling'] for result in results])

best_epochs_heating = results[best_index_heating]['Epochs']
best_epochs_cooling = results[best_index_cooling]['Epochs']

print(f"Best attempt for Heating Load: Epochs = {best_epochs_heating}")
print(f"Best attempt for Cooling Load: Epochs = {best_epochs_cooling}")

# Set font size for all plots
plt.rcParams.update({'font.size': 16})

# Heating Load Plot
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

# Cooling Load Plot
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

