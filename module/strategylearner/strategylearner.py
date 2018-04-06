# -*- coding: utf-8 -*-
"""
LSTMモデルで学習してみる
Kerasの勉強も兼ねてw
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

def buildModel(stone_max:int
               ,get_max:int
               ,lstm_num:int = 128
               ,dropout_rate:float = 0.5
               ,activation_func_name:str = "sigmoid"
               ,loss_func_name:str = "binary_crossentropy"
               ,optimize_name:str = "rmsprop"
               ,metrics_name:str = "accuracy"):
    """
    石の数と石を取ることができる数の最大値を入力のベースにする
    出力は取る石の数のため、出力層の数はget_max個になる
    """
    model = Sequential()
    model.add(Embedding(stone_max*get_max+1, output_dim = get_max + 1))
    model.add(LSTM(lstm_num))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation = activation_func_name))