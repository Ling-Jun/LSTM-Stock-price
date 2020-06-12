from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


def build(X, y, batch_size, epoch):
    # The LSTM architecture
    regressor = Sequential()
    # =======================================================================================================================
    # return_sequences: Whether to return the last output. in the output sequence, or the full sequence. Default: False.
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    # =======================================================================================================================
    regressor.add(Dense(units=1))
    # =======================================================================================================================
    regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
    print(regressor.summary())
    return regressor
