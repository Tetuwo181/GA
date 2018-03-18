#coding: UTF-8
import random
import numpy as np
from ...individual import individual as INDI
from ...individual import population as POP
from typing import Optional
from typing import Tuple
from numba import jit 


#ここでは選択の関数を書く
#実際にGAを回す際個体群を引数にとる形にインターフェースを統一する
#選択の引数はPopulationの派生クラスでなければならない



#確率に応じて選択。ランキング選択やルーレット選択から呼び出す
@jit
def select_from_prob(population, probs):
    prob_base = 0
    r = random.random()
    for individual, prob in zip(population[:], probs):
        if (prob_base < r) and (r < prob_base+prob):
            return individual
        else:
            prob_base = prob_base + prob
    return None


def from_rank(population):
    prob_base = np.array([index +1.0 for index in population.length])
    donominator = np.sum(prob_base)
    probs = ( (donominator-index)/donominator for index in prob_base )
    return select_from_prob(population.sorted(), probs)
                   

#ルーレット選択(phenotypeがlistである場合上手くいかない,つまり現状多目的最適化では工夫しないと使用できない)
#また、phenotypeが負の数を撮る時にも使えない
def roulette(population):
    phenotype_list = np.array([individual.phenotype for individual in population[:]])
    sum_of_phenotype = np.sum(phenotype_list)
    probs = (phenotype/sum_of_phenotype for phenotype in phenotype_list)
    return select_from_prob(population, probs)


#2個体から1個体を得るトーナメント
@jit    
def binary_tournament(individual1, individual2):
    if individual1 > individual2:
        return individual1
    else:
        return individual2

#トーナメントのベース
def tournament_base(population):
    if len(population) < 2:
        return population[0]
    if len(population) == 2:
        return binary_tournament(population[0], population[1])
    if len(population)%2 == 1:
        new_group = [population[-1]]
        individuals = population[:-1]
    else:
        new_group = []
        individuals = population[:]
    group1 = individuals[:len(individuals)/2]
    group2 = individuals[len(individuals)/2:]
    selected = [binary_tournament(individual1, individual2) for individual1, individual2
                 in zip(group1, group2)]
    new_group.extend(selected)
    return tournament_base(new_group)

#個体群から2個体をトーナメント
def set_tournament_num(tournament_num = 2):
    @jit
    def tournament(parents_pool):
        individuals = random.sample(parents_pool, tournament_num)
        return tournament_base(individuals)
    return tournament

def get_selection(selection_type = "RANK"):
    index_to_func = {}
    index_to_func["RANK"] = from_rank
    index_to_func["ROULETTE"] = roulette
    index_to_func["TOUR"] = set_tournament_num
    if selection_type in index_to_func:
        return index_to_func
    else:
        return from_rank
    
#個体群から親を選択する。返し値は親のリストのタプルを返すクロージャ
def set_algorithm(select, build_num = None):
    def exec_select(population):
        if build_num is None:
            couple_num = population.length
        else:
            couple_num = build_num
        fathers = INDI.Population([select(population) for index in range(couple_num)])
        mothers = INDI.Population([select(population) for index in range(couple_num)])
        return (fathers, mothers)
    return exec_select

#選択のクラス。上記の関数をコンストラクタに代入して実行
class Selection(object):
    def __init__(self, selection_algo, build_pop_num:Optional[int] = None):
        self.__algo = selection_algo
        self.__build_pop_num_default = build_pop_num
    
    def execute(self, population:INDI.Population) -> INDI.BaseIndividual:
        return self.__algo(population)
            
    def build_group(self, population:INDI.Population, choice_num:Optional[int] = None) -> INDI.Population:
        if choice_num is None:
            pop_num = population.length
        else:
            pop_num = choice_num
        return INDI.Population([self.execute(population) for index in range(pop_num)])
    
    def build_parents(self, population:INDI.Population, pair_num:int = 2, choice_num:Optional[int] = None):
        return tuple([self.build_group(population, choice_num) for index in range(pair_num)])
    


