# -*- coding: utf-8 -*-

from .test_problem import test_problem as TP
import individual as INDI

def test_init_individual():
    def geno_equal(genom1, genom2):
        for gen1, gen2 in zip(genom1, genom2):
            if gen1 is not gen2:
                return False
        return True
    test_genom1 = [0,0,1,1]
    test_genom2 = [1,1,1,0]
    test_incubator = INDI.incubator_builderincubator_builder(TP.binary_num, True, False)
    test_indi1 = test_incubator(test_genom1)
    test_indi2 = test_incubator(test_genom2)
    assert geno_equal(test_genom1, test_indi1.genotype)
    assert geno_equal(test_genom2, test_indi2.genotype)
    assert test_indi1.phenotype > test_indi2.phenotype
    

