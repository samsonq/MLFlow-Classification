import mlflow
import mlflow.keras

import numpy as np
import pandas as pd
import random

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def binary_model(dim):
    """
    Builds a simple binary classification neural network
    :param dim: dimensions of the input into model
    :return: binary classification model
    """
    mdl = Sequential([
        Dense(64, activation="relu", input_dim=dim),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])
    return mdl


def recall(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives/(possible_positives + K.epsilon())


def precision(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives/(predicted_positives + K.epsilon())


def f1(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val)/(precision_val + recall_val + K.epsilon()))


def train_and_log(mdl, train_x, train_y, valid_x, valid_y, p1, p2, p3):
    """
    Train and evaluate the model based on training and validation data and outputs the accuracy,
    recall, precision, and f1 score at each epoch. Logs inputted hyperparameters and metrics
    into MLFlow ui to visualize.
    :param mdl: model to train and log metrics
    :param train_x: training features
    :param train_y: training labels
    :param valid_x: validation features
    :param valid_y: validation labels
    :param p1: hyperparameter 1 (learning rate)
    :param p2: hyperparameter 2 (batch size)
    :param p3: hyperparameter 3 (epochs)
    :return: history of metrics from training
    """
    mdl.compile(optimizer=RMSprop(lr=p1),
                loss="binary_crossentropy",
                metrics=["accuracy", recall, precision, f1])

    with mlflow.start_run():
        hist = mdl.fit(train_x, train_y, batch_size=p2, epochs=p3,
                       verbose=1, validation_data=(valid_x, valid_y))
        mlflow.log_param("learning_rate", p1)
        mlflow.log_param("batch_size", p2)
        mlflow.log_param("epochs", p3)
        metrics = ["accuracy", "recall", "precision", "f1", "val_accuracy", "val_recall", "val_precision", "val_f1"]
        for metric in metrics:
            for step in range(len(hist.history[metric])):
                mlflow.log_metric(metric, hist.history[metric][step], step=step+1)
        mlflow.keras.log_model(mdl, "model")

    return hist.history


def preprocessing(df):
    """
    Preprocess and clean data to transform into training and validation sets.
    :param df: dataframe to prepare
    :returns: tensors of training and validation data
    """
    df.drop(labels=["Name", "Cabin", "Ticket"], axis=1, inplace=True)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    le_sex = LabelEncoder()
    le_sex.fit(df["Sex"])
    encoded_sex_training = le_sex.transform(df["Sex"])
    df["Sex"] = encoded_sex_training

    le_embarked = LabelEncoder()
    le_embarked.fit(df["Embarked"])
    encoded_embarked_training = le_embarked.transform(df["Embarked"])
    df["Embarked"] = encoded_embarked_training

    scale = StandardScaler()

    ages_train = np.array(df["Age"]).reshape(-1, 1)
    fares_train = np.array(df["Fare"]).reshape(-1, 1)
    df["Age"] = scale.fit_transform(ages_train)
    df["Fare"] = scale.fit_transform(fares_train)

    features = df.drop(labels=["PassengerId", "Survived"], axis=1)  # define training features set
    labels = df["Survived"]  # define training label set
    train_x, valid_x, train_y, valid_y = train_test_split(features, labels, test_size=0.2, random_state=0)
    return train_x.to_numpy(), valid_x.to_numpy(), train_y.to_numpy(), valid_y.to_numpy()


if __name__ == "__main__":

    data = pd.read_csv("data.csv")
    training_experiments = 20

    x_train, x_validation, y_train, y_validation = preprocessing(data)

    for iteration in range(1, training_experiments+1):
        print("Training iteration: " + str(iteration))
        P1 = random.uniform(0.000001, 0.1)   # learning rate
        P2 = random.randint(10, 500)   # batch size
        P3 = random.randint(10, 50)     # epochs

        model = binary_model(x_train.shape[1])
        history = train_and_log(model, x_train, y_train, x_validation, y_validation, P1, P2, P3)
        print(history)
        print("Training iteration " + str(iteration) + " completed.")
