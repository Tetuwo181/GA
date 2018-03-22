# -*- coding: utf-8 -*-
from ....indi import individual as INDI
from ....indi import population as POP
from ....test_problem import test_problem as TP


"""
テストに使う個体を初期化
テスト問題は遺伝子中の１の数をカウント
"""
incubator = INDI.incubator_builder(TP.binary_num, True)
test_indi_bin1 = incubator([0, 0, 0, 0, 0])
test_indi_bin2 = incubator([1, 0, 0, 0, 0])
test_indi_bin3 = incubator([1, 1, 0, 0, 0])
test_indi_bin4 = incubator([1, 0, 0, 1, 0])
test_indi_bin5 = incubator([1, 1, 1, 1, 0])
test_indi_bin6 = incubator([1, 1, 1, 1, 1])

test_population_bin = POP.Population([test_indi_bin1, test_indi_bin2, test_indi_bin3, test_indi_bin4, test_indi_bin5])


