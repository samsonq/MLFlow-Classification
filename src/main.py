import sys
import os

import numpy as np
import pandas as pd
import random
import argparse

import mlflow
import mlflow.keras

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def binary_model(dim):
    """
    Builds a simple binary classification neural network.
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
    Calculates recall metric through false positive rates based on
    true values and predictions.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: recall score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives/(possible_positives + K.epsilon())


def precision(y_true, y_pred):
    """
    Calculates precision metric through false negative rates based on
    true values and predictions.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: precision score
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives/(predicted_positives + K.epsilon())


def f1(y_true, y_pred):
    """
    Calculates f1 score based on the precision and recall metrics of the
    true values and predictions.
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: f1 score
    """
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val)/(precision_val + recall_val + K.epsilon()))


def train_and_log(mdl, train_x, train_y, valid_x, valid_y, learning_rate, batch_size, epochs):
    """
    Train and evaluate the model based on training and validation data and outputs the accuracy,
    recall, precision, and f1 score at each epoch. Logs inputted hyperparameters and metrics
    into MLFlow ui to visualize.
    :param mdl: model to train and log metrics
    :param train_x: training features
    :param train_y: training labels
    :param valid_x: validation features
    :param valid_y: validation labels
    :param learning_rate: gradient descent learning rate
    :param batch_size: training batch size
    :param epochs: number of epochs
    :return: history of metrics from training
    """
    mdl.compile(optimizer=RMSprop(lr=learning_rate),
                loss="binary_crossentropy",
                metrics=["accuracy", recall, precision, f1])

    with mlflow.start_run():
        hist = mdl.fit(train_x, train_y, batch_size=batch_size, epochs=epochs,
                       verbose=1, validation_data=(valid_x, valid_y))
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        metrics = ["acc", "recall", "precision", "f1", "val_acc", "val_recall", "val_precision", "val_f1"]
        for metric in metrics:
            for step in range(len(hist.history[metric])):
                mlflow.log_metric(metric, hist.history[metric][step], step=step+1)
        mlflow.keras.log_model(mdl, "model")

    return hist.history


def preprocess(df):
    """
    Preprocess and clean titanic data to transform into training and validation sets.
    :param df: titanic dataframe to prepare
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


def parse_args():
    """
    Get arguments to run detection model. Input video file to detect social distancing, or
    by default, use webcam for detection.
    :return: program arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-lr", "--learning_rate", type=float, default="0.0001",
                    help="model learning rate")
    ap.add_argument("-b", "--batch_size", type=int, default="64",
                    help="training batch size")
    ap.add_argument("-e", "--epochs", type=int, default="50",
                    help="training epochs")
    return vars(ap.parse_args())


if __name__ == "__main__":

    data = pd.read_csv("../data/titanic.csv")
    training_experiments = 20
    args = parse_args()

    x_train, x_validation, y_train, y_validation = preprocess(data)

    for iteration in range(training_experiments):
        print("Training experiment: " + str(iteration+1))

        LEARNING_RATE = args["learning_rate"]
        BATCH_SIZE = args["batch_size"]
        EPOCHS = args["epochs"]

        model = binary_model(x_train.shape[1])
        history = train_and_log(model, x_train, y_train, x_validation, y_validation,
                                LEARNING_RATE, BATCH_SIZE, EPOCHS)
        print(history)
        print("Training experiment " + str(iteration+1) + " completed.")
