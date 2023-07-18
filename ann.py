# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 05:56:01 2023

@author: aitza
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import scikitplot as skplt

# Import dataset
dataset = pd.read_csv('Satellite_3.csv')

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the Artificial Neural Network (ANN) model
ann_model = tf.keras.models.Sequential()
# Adding the input layer
ann_model.add(tf.keras.layers.Dense(units=64, activation='relu'))
# Adding additional hidden layers
ann_model.add(tf.keras.layers.Dense(units=32, activation='relu'))
ann_model.add(tf.keras.layers.Dense(units=16, activation='relu'))
# Adding the output layer
ann_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN model
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN model
ann_model.fit(X_train, y_train, batch_size=32, epochs=10)

# Prediction
y_pred = ann_model.predict(X_test)
y_pred = (y_pred > 0.5)

# Accuracy and Confusion Matrix
confusion_mat = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Accuracy score is: ", accuracy, "%")

# Normalized Confusion Matrix
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
