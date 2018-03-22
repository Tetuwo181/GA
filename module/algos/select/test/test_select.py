# -*- coding: utf-8 -*-
from ....indi import individual as INDI
from ....indi import population as POP
from ....test_problem import test_problem as TP
from module.algos.select import select as  SL




def test_binary_rournament():
    choiced = SL.binary_tournament(test_indi1, test_indi2)
    assert choiced == test_indi2
    choiced = SL.binary_tournament(test_indi3, test_indi5)
    assert choiced == test_indi5


    