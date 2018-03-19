# -*- coding: utf-8 -*-
import numpy as np
import csv
from collections import namedtuple
Item = namedtuple('Item', ('weight', 'price'))

def binary_num(gene):
    vector_of_gene = np.array(gene)
    return np.sum(vector_of_gene)

def binary_two(gene):
    """
    多目的最適化のテスト用
    遺伝子中の1の数と0の数をカウントし、そのタプルを返す
    """
    num_zero = 0
    num_one = 0
    for gen in gene:
        if gen == 0:
            num_zero = num_zero + 1
        else:
            num_one = num_one + 1
    return (num_zero, num_one)


def init_knapsack(item_set, capacity):
    """
    ナップサック問題の初期化。
    アイテムと容量を入れて初期化
    返し値はナップサック問題を実行する関数
    """
    def knapsack_problem(gene):
        """
        ナップサック問題
        アイテムが容量を上回る場合は入れないようにする
        """
        price = 0
        weight = 0
        for item, will_get in zip(item_set, gene):
            if weight + item.weight*will_get < capacity:
                price = price + item.price
                weight = weight + item.weight
        return price
    return knapsack_problem


def init_items(item_path):
    """
    ナップサック問題のアイテムの量と容量を初期化する
    返し値はアイテムのリストとナップサックのキャパシティ
    """
    with open(item_path, "r") as item_file:
        reader = csv.reader(item_file)
        capacity = int(reader[0])
        item_set = [Item(int(data[0]), int(data[1])) for data in reader[1:]]
    return (capacity, item_set)
        