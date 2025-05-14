# ⚡ DeepEnergy: Energy Consumption Prediction in Residential Buildings Using Deep Learning in the Qassim Region

This project focuses on predicting heating and cooling energy loads in residential buildings located in the Qassim region, using various building characteristics as input features.
By applying deep learning models, the project aims to estimate energy loads accurately and support smarter building design decisions. It also contributes to more efficient energy planning and sustainability in the residential sector.

---

## 📌 Project Objectives

- Build and compare multiple deep learning models (DNN, CNN, LSTM, CNN-LSTM) to predict energy consumption.
- Use a dataset containing building characteristics from residential units in the Qassim region.
- Evaluate models using standard performance metrics (MSE, MAE, RMSE, R²).
- Identify the most accurate model and analyze the effect of batch size on performance.
- Support energy-efficient planning and smart building design decisions.

---

## 📊 Dataset Description

The dataset contains 3,833 samples of residential buildings, with each sample representing a unique building design.

### Input features include:
- Building Area (m²)
- Floor Height (m)
- Exterior Window Area (m²)
- Opaque Exterior Wall Area (m²)
- Window-to-Wall Ratio (WWR %)
- Window U-value (W/m²K)
- Roof U-value (W/m²K)
- Wall U-value (W/m²K)

### Target variables:
- **Cooling Load** (kWh/m².yr)
- **Heating Load** (kWh/m².yr)

---

## 🧠 Models Used

The following deep learning models were implemented and compared:

- **DNN**: A basic deep neural network using fully connected layers.
- **CNN**: A convolutional neural network that captures patterns across input features.
- **LSTM**: A recurrent model used to explore its performance in learning feature relationships.
- **CNN-LSTM**: A hybrid model that combines convolutional layers with LSTM units to enhance prediction accuracy.

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Google Colab

---

## 📈 Results Summary

- Four deep learning models (DNN, CNN, LSTM, CNN-LSTM) were trained and evaluated using MSE, RMSE, MAE, and R².
- CNN-LSTM with batch size 64 achieved the best performance, with an R² score above 0.99 for both heating and cooling.
- Increasing the batch size generally improved model performance.
- Scatter and violin plots were used to compare actual vs. predicted values across all models.

---

## 👩‍💻 Contributors

This project was completed as part of the Graduation Project – Phase 2  
Department of Information Technology, Qassim University.

- Khozama Alnasyan  
- Joury Alghofaily  
- Cady Alali

Supervised by Dr. Dina M. Ibrahim
