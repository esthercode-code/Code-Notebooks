import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# 1. Feature Engineering: Creating Sequences for LSTM
def create_sequences(data, time_steps=50):
    """Transforms tabular data into 3D sequences for LSTM (Samples, TimeSteps, Features)"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 1:])  # All sensor features
        y.append(data[i + time_steps, 0])  # The failure label
    return np.array(X), np.array(y)


# 2. Model Architecture
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Probability of failure
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


# 3. Training Logic (Simplified)
# X_train_seq has shape (Samples, 50, Number_of_Features)
model = build_lstm_model((50, X_train.shape[1]))
history = model.fit(X_train_seq, y_train, epochs=20, batch_size=32, validation_split=0.1)