# ============================================================================================
# PROBLEM C5
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model 
# should be trained to predict the next 24 observations of the 7 variables.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
#
# Desired MAE < 0.1 on the normalized dataset.
# ============================================================================================

# ============================================================================================
# PROBLEM C5
#
# Build and train a neural network to predict time indexed variables of
# the multivariate house hold electric power consumption time series dataset.
# Using a window of past 24 observations of the 7 variables, the model 
# should be trained to predict the next 24 observations of the 7 variables.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
#
# Desired MAE < 0.1 on the normalized dataset.
# ============================================================================================

import urllib
import os
import zipfile
import pandas as pd
import tensorflow as tf

def download_and_extract_data():
    url = 'https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/household_power.zip'
    urllib.request.urlretrieve(url, 'household_power.zip')
    with zipfile.ZipFile('household_power.zip', 'r') as zip_ref:
        zip_ref.extractall()

def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

def windowed_dataset(series, batch_size, n_past=24, n_future=24, shift=1):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(n_past + n_future))
    dataset = dataset.shuffle(1000)
    dataset = dataset.map(lambda window: (window[:-n_future], window[-n_future:, :1]))
    
    return dataset.batch(batch_size).prefetch(1)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae')<0.1):
      print("\nMAE < 0.1!")
      self.model.stop_training = True

def solution_C5():
    download_and_extract_data()
    df = pd.read_csv('household_power_consumption.csv', sep=',',
                     infer_datetime_format=True, index_col='datetime', header=0)

    N_FEATURES = df.shape[1]

    data = df.values
    split_time = int(len(data) * 0.5)
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    x_train = data[:split_time]
    x_valid = data[split_time:]

    BATCH_SIZE = 32
    N_PAST = 24
    N_FUTURE = 24
    SHIFT = 1

    train_set = windowed_dataset(x_train, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)
    valid_set = windowed_dataset(x_valid, BATCH_SIZE, N_PAST, N_FUTURE, SHIFT)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=[N_PAST, N_FEATURES], activation="relu"), 
        tf.keras.layers.Dense(N_FUTURE),
    ])

    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), metrics=["mae"])
    callbacks = myCallback()
    history = model.fit(train_set, epochs=100, validation_data=valid_set, callbacks=[callbacks])

    return model

if __name__ == '__main__':
    model = solution_C5()
    model.save("model_C5.h5")