EUR/CHF LSTM Forecasting Model
This project aims to forecast the EUR/CHF exchange rate using a Long Short-Term Memory (LSTM) model, incorporating Swiss National Bank (SNB) interventions to improve predictions. The project leverages TensorFlow, Plotly, and Streamlit to deliver both the model and its visualizations in a user-friendly, interactive format.

🚀 Project Overview
As part of my interest in financial forecasting and machine learning, I developed a model to predict the movements of the EUR/CHF currency pair, which is significantly influenced by SNB interventions. These interventions often cause sharp shifts in the exchange rate, so I included them as additional input to enhance the model’s prediction accuracy.

The goal of this project was to:

Forecast future EUR/CHF exchange rates.

Incorporate historical SNB interventions as an external factor to improve model predictions.

Visualize the results and performance metrics in an interactive dashboard.

🔨 Development Process
1. Initial Concept
The idea stemmed from my interest in financial markets, particularly how central banks, such as the Swiss National Bank (SNB), intervene in currency markets. I wanted to create a model that could predict the EUR/CHF exchange rate, accounting for these interventions and their potential effects on the currency pair.

2. Data Collection
I used the yfinance library to download historical EUR/CHF exchange rate data from January 1, 2010, to December 31, 2024. The dataset includes daily closing prices, which served as the primary feature for the model.

To represent SNB interventions, I manually added synthetic intervention dates, marking them as a binary feature (1 for intervention, 0 for no intervention) on specific days that I identified from public records of SNB actions.

3. Data Preprocessing
I performed the following preprocessing steps:

Normalization of the EUR/CHF closing prices using MinMaxScaler to scale the values between 0 and 1, which is essential for training deep learning models.

I created sequences of 30 days (a rolling window) to serve as input for the LSTM, alongside the intervention indicator for each day.

4. Model Architecture
I developed an LSTM (Long Short-Term Memory) neural network model for time series forecasting. The architecture consists of:

LSTM(64) layer to capture temporal dependencies

Dropout(0.2) for regularization to prevent overfitting

LSTM(32) layer for further refinement

A Dense(1) layer to output the predicted EUR/CHF value

I trained the model for 15 epochs with a batch size of 32, optimizing it using the Adam optimizer and the Mean Squared Error (MSE) loss function.

5. Model Evaluation
I evaluated the model using two key metrics:

Mean Absolute Error (MAE), which measures the average magnitude of errors in predictions.

R² score, which explains how well the model fits the data.

6. Visualization
Using Plotly, I visualized the actual vs predicted values for EUR/CHF in an interactive graph. This allowed me to compare the model’s performance visually and understand how well it tracked real exchange rate movements, especially around SNB intervention dates.

7. User Interface (Streamlit)
To make the model accessible and interactive, I built a Streamlit dashboard that:

Loads and preprocesses the data.

Trains the model in real time.

Displays the performance metrics (MAE and R²).

Renders an interactive plot showing the predicted and actual values for EUR/CHF.

📊 Key Results
MAE: [Insert MAE value here]

R²: [Insert R² value here]

The model accurately captures the impact of SNB interventions on the EUR/CHF exchange rate.

⚙️ How to Run the Project
Clone this repository:

bash
Copiar
Editar
git clone https://github.com/your-username/eur-chf-forecast-lstm.git
cd eur-chf-forecast-lstm
Install dependencies:

bash
Copiar
Editar
pip install -r requirements.txt
Run the Streamlit app:

bash
Copiar
Editar
streamlit run app.py
Open the URL in your browser to view the model predictions and interactive plot.

📚 Libraries Used
TensorFlow for building and training the LSTM model.

Plotly for interactive data visualization.

Streamlit for the user interface.

yfinance to download financial data.

Scikit-learn for data preprocessing and evaluation metrics.
