# 📈 Stock Predictions Using LSTM

This repository contains a deep learning-based stock price prediction model using **Long Short-Term Memory (LSTM) networks**. The model is trained on historical stock data and attempts to predict future closing prices. The project is implemented in Python using **TensorFlow/Keras**, **pandas**, and **scikit-learn**.

## 🚀 Features

- 📊 **Historical Data Analysis** – Fetches stock data using Yahoo Finance.
- 🔄 **Data Preprocessing** – Normalizes and structures data for training.
- 🧠 **LSTM Model** – Uses a deep learning approach for time-series forecasting.
- 📉 **Stock Price Predictions** – Predicts future stock closing prices.
- 📈 **Visualization** – Graphical representation of actual vs. predicted prices.

## 📦 Installation

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/arman-dogru/stock-predictions-regression.git
cd stock-predictions-regression
```

### **2️⃣ Install Dependencies**
Ensure you have Python 3 installed, then install the required libraries:
```sh
pip install pandas numpy keras tensorflow sklearn matplotlib pandas_datareader
```

### **3️⃣ Run the Notebook**
Execute the Jupyter Notebook containing the model:
```sh
jupyter notebook Untitled.ipynb
```

## 📌 How It Works

1. **Data Fetching**:
   - Stock data is retrieved from Yahoo Finance using `pandas_datareader`.
   - The dataset includes Open, High, Low, Close, Volume, and Adjusted Close prices.

2. **Preprocessing**:
   - Only the `Close` column is used for prediction.
   - Data is normalized using **MinMaxScaler** for better training efficiency.
   - The dataset is split into **80% training data** and **20% testing data**.

3. **Training the Model**:
   - The LSTM model is built using **Keras Sequential API**.
   - A sequence of past **60 days** is used to predict the **next day's closing price**.
   - Model layers include **LSTM**, **Dense (fully connected layers)**, and **ReLU activation functions**.

4. **Prediction & Evaluation**:
   - The trained model predicts stock prices based on test data.
   - Results are compared against actual closing prices.
   - Root Mean Squared Error (RMSE) is computed to evaluate performance.

## 📷 Results & Visualization

Below is a comparison of the **actual vs. predicted** stock prices:

| Training Phase | Prediction Phase |
|---------------|-----------------|
| ![Training](imgs/training.png) | ![Predictions](imgs/prediction.png) |

## 📊 Sample Output

The model predicts stock prices as follows:
```plaintext
Predicted Price: $41,536.73
Actual Price: $46,202.14
```
(Note: Prices vary based on stock symbol and date range.)

## 📂 Files & Directories

- **Untitled.ipynb** – Jupyter Notebook containing the full implementation.
- **predictions2011-2019.html/pdf** – Model output for predictions between 2011-2019.
- **predictions2011-2022.html/pdf** – Model output for predictions between 2011-2022.
- **predictionsBTC-USD(50).html** – Bitcoin price prediction results.

## 🏗 Future Improvements

- Enhance model accuracy using additional technical indicators (e.g., RSI, MACD).
- Experiment with different architectures (GRU, Transformer models).
- Implement real-time stock prediction using live market data.

## 🤝 Contributing

Contributions are welcome! If you would like to improve the model, add features, or fix issues, please fork the repository and submit a pull request.

## 📜 License

This project is open-source and available under the **MIT License**.

---

### ⭐ **If you found this useful, give it a star! 🌟**
