# 📈 Stock Price Prediction Using LSTM Networks

## 🧠 Overview

This project demonstrates how to predict stock prices using **Long Short-Term Memory (LSTM)** networks — a type of recurrent neural network (RNN) well-suited for time series data. Leveraging **TensorFlow** and **Keras**, the model forecasts stock prices based on historical data, supporting more informed, data-driven investment insights.

---

## ✨ Key Features

- 📊 **Data Acquisition**  
  Retrieves historical stock data using the `yfinance` library.

- 🧹 **Data Preprocessing**  
  Cleans and scales the dataset using `MinMaxScaler` and handles missing values efficiently.

- 🏗️ **Model Architecture**  
  Implements a deep LSTM-based neural network using TensorFlow/Keras.

- 📉 **Visualisation**  
  Provides detailed plots of stock trends and prediction performance for easier interpretation.

---

## ⚙️ Getting Started

### 🔧 Prerequisites

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn tensorflow yfinance scikit-learn
```
## 📁 Project Structure

```bash
├── data/
│   └── all_stocks_5yr.csv
├── notebooks/
│   └── main.ipynb
├── images/
├── src/
│   └── modules/
│       └── Modules.py
├── README.md
└── requirements.yml
```

## 📊 Stock Analysis

Let us begin with a quick overview of some of the most well-known and influential technology companies — including NVIDIA, Google, Apple, Facebook, and others. The figure below illustrates the trend of their **open** and **close** stock prices over a five-year period:

![Image](https://github.com/user-attachments/assets/2bec4206-e021-49fb-9c8f-bb9618ab5d9a)

---

Next, we visualise the **trading volume** of these nine stocks over time:

![Image](https://github.com/user-attachments/assets/a7745c1c-45d9-4492-95b0-169c2456b363)

### 📈 What Does Trading Volume Indicate?

In the stock market, **volume** refers to the number of shares or contracts traded for a particular security within a given timeframe (typically a trading day). It reflects the level of trading activity and serves as a key metric in understanding market behaviour:

- **High volume** indicates strong interest and liquidity in a stock.
- **Low volume** implies reduced activity or investor interest.

Volume patterns often help investors confirm price trends or anticipate reversals, making it a vital component in technical analysis.

---

In the following section, we will perform a more detailed analysis of **Apple’s stock**, viewed as a standalone case study.



## 🍏 Apple Stock Price Analysis (2013–2018)
This section highlights a visual analysis of Apple Inc. (AAPL) stock prices between 2013 and 2018. The chart below illustrates key trends and price movements, offering insight into market behaviour over this five-year span.


![Image](https://github.com/user-attachments/assets/a5a7f4f5-f985-4d78-af30-7418cf329aff)


## 🔍 Key Observations

### 📈 Overall Upward Trend
The stock price steadily increased, reflecting strong growth in Apple’s market value.

### ⚖️ Volatility Within Growth
- **2013–2015:** Consistent upward movement.
- **2015–2016:** Short-term correction or consolidation, with a dip in early 2016.
- **2016–2018:** Renewed strong growth, reaching all-time highs.

### 💰 Estimated Price Range
- **Start:** ~$60–70 (2013)
- **Mid:** ~$130–135 (2015)
- **Dip:** ~$90–100 (2016)
- **Peak:** ~$180+ (2018)

---

## 🧱 Model Architecture & Training Summary

The LSTM model architecture consists of:

- 🔄 Two stacked `LSTM` layers (64 units each)
- 🎯 A `Dense` layer (32 units)
- 🛡️ A `Dropout` layer (0.5) to prevent overfitting
- 🔚 A final `Dense` output layer (1 unit) for closing price prediction

### 🧾 Model Summary
- **Total trainable parameters:** 52,033  
- **Input shape:** 60 time steps × 1 feature  
- **Loss function:** Mean Squared Error (MSE)  
- **Optimiser:** Adam

---

### 📉 Training Performance (10 Epochs)
- **Initial loss:** `0.0586`
- **Final loss:** `0.0063`

The consistent drop in loss shows that the model effectively learned from historical data, improving accuracy with each epoch.

---

## 📏 Evaluation Metrics

- **Mean Squared Error (MSE):** `53.57`
- **Root Mean Squared Error (RMSE):** `7.32`

🔍 These values reflect a **moderate prediction error**, meaning the model’s forecasted stock prices deviate by roughly **$7.32 on average** from actual prices — a reasonable outcome given market volatility.

---

## 📊 Prediction Model Analysis

The chart below visualises the model's performance on unseen test data:

![Image](https://github.com/user-attachments/assets/7158ccb2-6b29-4ca3-909f-852df7d18265)

- 🔵 **Training Data:** Historical prices used to fit the model  
- 🟠 **Test Data:** Real stock prices the model hasn’t seen  
- 🟢 **Predictions:** Forecasted prices by the LSTM model

### 🔍 Performance Insights
- 📌 **Trend Detection:** The model accurately follows the general upward direction of the stock.
- 🧘 **Stable Periods:** Predictions align well in steady market conditions.
- ⚠️ **Volatility Challenges:**  
  Struggles slightly during abrupt rises/falls (e.g. early 2018), where it tends to *smooth out sharp changes*.  
  This is a typical limitation of LSTM models dealing with noisy financial data.

---

## ✅ Conclusions

- ✅ Successfully demonstrates time series forecasting with LSTM networks.
- 📈 Effectively captures long-term stock price trends.
- ⚠️ Limitations exist in modelling short-term volatility and sharp price swings.

---

## 📦 Dependencies

This project uses the following Python libraries:

| Library           | Purpose                                        |
|-------------------|------------------------------------------------|
| `pandas`          | Data handling and manipulation                 |
| `numpy`           | Numerical computations                         |
| `matplotlib`      | Static visualisation                           |
| `seaborn`         | Enhanced plotting (optional)                   |
| `yfinance`        | Stock data from Yahoo Finance (optional)       |
| `scikit-learn`    | Data preprocessing (e.g., `MinMaxScaler`)      |
| `tensorflow/keras`| Deep learning model development                |

---

## 🚀 Future Enhancements

To improve accuracy and applicability:

- 🧠 **Advanced Models**: Try GRU, ARIMA, or hybrid models with attention.
- ⚙️ **Hyperparameter Tuning**: Use grid/random search for better results.
- 📈 **New Features**: Add technical indicators (RSI, MACD), volume, or sentiment.
- 🌀 **Rolling Forecasting**: Use a moving window for continuous predictions.
- 📏 **More Metrics**: Evaluate with MAE, MAPE, and others.
- 📊 **Interactive Visuals**: Integrate with Plotly or Streamlit.
- 🌐 **Web Deployment**: Deploy via Streamlit or Flask for interactive use.

---

## 👨‍💻 Author

Developed with ❤️ by **Alberto AJ**, AI/ML Engineer  
📌 [Visit my GitHub](https://github.com/AlbertoGhub/AlbertoGhub)
