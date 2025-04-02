import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #remover as mensagens do tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras
import pandas as pd
import sklearn
import numpy as np

from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import InputLayer, Dense #type:ignore
from tensorflow.keras import utils as np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 

# Classe de previsores 

base = pd.read_csv('personagens.csv')

X = base.iloc[:,0:6].values
print(X)

y = base.iloc[:,6].values
print(y)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

y = np_utils.to_categorical(y)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, train_size=0.25)



redeNeural = Sequential()

redeNeural.add(InputLayer(shape = 6,))
redeNeural.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'random_uniform'))
redeNeural.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'random_uniform'))
redeNeural.add(Dense(units = 2, activation = 'softmax'))

redeNeural.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

redeNeural.fit(X_treinamento, y_treinamento, batch_size = 10, epochs = 5)
