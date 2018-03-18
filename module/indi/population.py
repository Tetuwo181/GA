# -*- coding: utf-8 -*-
import numpy as np
import random
import module.indi.individual as INDI
from typing import List
from typing import Callable

class Population(object):
    """
    個体群を管理するクラス
    """
    def __init__(self, individuals:List[INDI.BaseIndividual]):
        self.__individuals = individuals[:]
        self.__length = len(self.__individuals)
        self.__is_rank_setted = False

    def __getitem__(self, index) -> List[INDI.BaseIndividual]:
        return self.__individuals[index]

    @property
    def individuals(self)-> List[INDI.BaseIndividual]:
        return self.__individuals

    @property
    def sorted_population(self):         
        return Population(sorted(self.individuals))     

    @property
    def length(self):
        return self.__length
    
    def set_rank(self):
        if self.__is_rank_setted:
            return
        for individual in self.individuals:
            individual.set_rank(self.individuals)

    def merge(self, others):
        new_individuals = self.individuals[:]
        new_individuals.extend(others[:])
        return Population(new_individuals)

    def get_superior(self, superior_num:int):
        sorted_individuals = sorted(self.individuals)
        return Population(sorted_individuals[:superior_num])

    def elite(self, upper:int = 25):
        if upper > 100:
            return self.sorted_population
        individual_num = int(self.length*upper/100)
        return Population(self.sorted_population[:individual_num])

    def elite_and_not(self, upper:int = 25):
        if upper > 100:
            return (self.sorted_population, None)
        else:
            elites = self.elite(upper)
            return (elites, Population(self.sorted_population[elites.length:]))
    
    def get_rank_population(self, rank = 1):
        if self.__is_rank_setted is False:
            self.set_rank()
        rank_group = []
        for index in range(rank):
            rank_group.append(Population([individual for individual
                                          in self.individuals
                                          if individual.rank == index]))
        return rank_group


def population_initializer(pop_num:int, genotype:str, gene_length:int, indi_generator:Callable[[INDI.Locus], INDI.Phenotype]):
    def init_real(problem:Callable[[INDI.Locus], INDI.Phenotype], max_list:np.ndarray = None, min_list:np.ndarray = None) -> Population:
        if max_list is None:
            maxes = np.array([1.0 for index in range(gene_length)])
        else:
            maxes = max_list
        if min_list is None:
            mins = np.array([0 for index in range(gene_length)])
        else:
            mins = min_list
        ranges = maxes - mins
        rand_bases = np.random.rand(pop_num, gene_length)
        return Population([indi_generator(rand_base*ranges, problem) for rand_base in rand_bases])

    def init_bin(problem:Callable[[INDI.Locus], INDI.Phenotype]) -> Population:
        return Population([indi_generator([random.randint(0, 1) for index_gene in range(gene_length)], problem) for index_indi in range(pop_num)])

    def init_order(problem:Callable[[INDI.Locus], INDI.Phenotype],) -> Population:
        indexes = [index for index in range(gene_length)]
        return Population(indi_generator(random.sample(indexes, gene_length), problem) for index_indi in range(pop_num))

    genotype_to_algo = {}
    genotype_to_algo["BIN"] = init_bin
    genotype_to_algo["ORD"] = init_order
    genotype_to_algo["REAL"] = init_real
    if genotype not in genotype_to_algo:
        return None
    else:
        return genotype_to_algo[genotype] 

def population_builder(genom_set, problem, indi_builder):
    return Population([indi_builder(genom, problem) for genom in genom_set])


