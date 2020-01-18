import mlflow
from mlflow import log_metric, log_param, log_artifact

import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


X_train = np.array([[1, 2, 3], [2, 4, 1], [4, 2, 1], [5, 3, 2]])
y_train = np.array([1, 0, 1, 0])


model = Sequential([
    Dense(64, activation="relu", input_shape=(3,)),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

P1 = 0.001

model.compile(optimizer=RMSprop(lr=P1),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=1, epochs=5, verbose=1)


if __name__ == "__main__":
    pass
