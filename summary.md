Data Handling & Preprocessing

The app fetches historical price data from Yahoo Finance or CSV.

Data must include Date and Close prices; Open, High, Low, Volume are optional.

Dates are converted to a proper datetime format.

Data is sorted chronologically to maintain time sequence.

Any missing or NaN values are removed.

The Close prices are normalized to a range of 0–1 using MinMaxScaler.

Normalization helps the neural network train faster and avoid large numerical errors.

A lookback window (e.g., 60 days) is chosen to feed past data into the model.

Creating Input & Output Sequences

Each training example consists of lookback previous days of Close prices.

The target value is the Close price immediately after the lookback window.

This creates X (inputs) and y (outputs) arrays for training the LSTM.

Inputs are reshaped into 3D arrays [samples, timesteps, features] for LSTM.

This captures the sequential, time-dependent nature of the data.

LSTM Model Logic

LSTM is a type of recurrent neural network specialized for sequences.

It remembers patterns across long sequences using memory cells.

Each LSTM cell uses gates to decide what to keep, forget, or output.

Dropout layers are added to prevent overfitting by randomly ignoring neurons.

The final Dense layer outputs a single predicted price for the next day.

The model is trained to minimize Mean Squared Error (MSE) between predicted and actual prices.

Training & Optimization

Data is split into training (80%) and testing (20%) sets.

The model trains over multiple epochs, repeatedly updating weights.

The Adam optimizer adjusts weights to reduce MSE efficiently.

Training progress is monitored in real time using a Streamlit callback.

Validation split (10%) checks for overfitting during training.

Evaluation Metrics

RMSE measures average prediction error magnitude.

MAE measures average absolute error.

MAPE measures prediction error in percentage terms.

These metrics quantify how well the model predicts unseen data.

Prediction & Forecasting

After training, the model predicts the test set and future prediction_days.

Future predictions are generated recursively: the last predicted value becomes part of the input for the next prediction.
