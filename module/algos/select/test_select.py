# -*- coding: utf-8 -*-
from ...indi import individual as INDI
from ...indi import population as POP
from ...test_problem import test_problem as TP
from module.algos.select import select as  SL

"""
テストに使う個体を初期化
テスト問題は遺伝子中の１の数をカウント
"""
incubator = INDI.incubator_builder(TP.binary_num, True, False)
test_indi1 = incubator([0, 0, 0, 0, 0])
test_indi2 = incubator([1, 0, 0, 0, 0])
test_indi3 = incubator([1, 1, 1, 0, 0])
test_indi4 = incubator([1, 0, 0, 1, 0])
test_indi5 = incubator([1, 1, 1, 1, 1])
test_indi6 = incubator([1, 1, 1, 1, 0])

test_population = POP.Population([test_indi1, test_indi2, test_indi3, test_indi4, test_indi5])

def test_binary_rournament():
    choiced = SL.binary_tournament(test_indi1, test_indi2)
    assert choiced == test_indi2
    choiced = SL.binary_tournament(test_indi3, test_indi5)
    assert choiced == test_indi5


    