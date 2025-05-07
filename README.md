🧠 Project Idea and Development Process
The motivation behind this project was to forecast the EUR/CHF exchange rate using deep learning, while incorporating macro-financial events — specifically, interventions by the Swiss National Bank (SNB) — as additional signals in the model. Here's a breakdown of how the idea evolved and how I implemented the solution:

🔹 1. Initial Concept
As part of my interest in both machine learning and financial markets, I was particularly intrigued by the role of central bank interventions on currency pairs. The EUR/CHF pair is significantly affected by the SNB, so I aimed to model those effects using an LSTM neural network.

🔹 2. Data Collection
I used the yfinance library to retrieve historical EUR/CHF exchange rate data from 2010 to 2024. Then, I manually created a small dataset of synthetic SNB intervention dates, marking them with a binary feature.

🔹 3. Feature Engineering
For each time step in the series, I included:

The normalized EUR/CHF close price

A binary variable indicating whether an SNB intervention occurred that day

I used a rolling window approach (30 days) to create sequences that serve as input for the LSTM, which learns patterns from historical price behavior and intervention contexts.

🔹 4. Model Architecture
I built a 2-layer LSTM model using TensorFlow/Keras, with dropout regularization to prevent overfitting. The model architecture was:

LSTM(64) with return_sequences=True

Dropout(0.2)

LSTM(32)

Dense(1) to predict the next day's price

🔹 5. Training and Evaluation
The model was trained on 80% of the data and tested on the remaining 20%. Evaluation metrics included:

MAE (Mean Absolute Error)

R² score for goodness-of-fit

🔹 6. Visualization
I used Plotly to create an interactive chart comparing the actual vs predicted EUR/CHF values over time. This helps visually assess how well the model captured price dynamics.

🔹 7. User Interface
To make the project more accessible and visually appealing, I built a dashboard using Streamlit, which:

Loads and preprocesses the data

Trains the model in real time

Displays performance metrics

Renders the forecast plot interactively

This project demonstrates how combining domain knowledge (SNB interventions) with machine learning techniques (LSTM) can enhance forecasting models in financial applications.
