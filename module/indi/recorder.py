# -*- coding: utf-8 -*-

from matplotlib import pyplot
import module.indi.population as POP
from abc import ABC


class Recorder(ABC):
    """
    各世代で優秀であった個体を記録するクラスの抽象クラス
    一目的と多目的でそれぞれ継承して使う
    """
    def __init__(self, record_num = 1):
        self.__record_num = record_num
        
    def record(self, population:POP.Population):
        pass

    def get_record_of_generation(self, generation):
        return self.__population(generation)
    
    @property
    def record_num(self):
        return self.__record_num


class RecorderSingle(Recorder):
    """
    1目的のレコーダー
    コンストラクタで上位n個体を記録するように設定する
    """
    def __init__(self, record_num = 1):
        super.__init__(record_num)
        self.__tops_of_generation = []
    
    def record(self, population:POP.Population):
        self.__tops_of_generation.append(population.sorted_population[:self.record_num])
        
    def top_individual(self, generation):
        return self.__tops_of_generation[generation]
    
    @property
    def get_top_phenotypes(self):
        phenotypes = [individuals[0].phenotype for individuals
                      in self.__tops_of_generation]
        generations = [index for index in range(len(phenotypes))]
        return (generations, phenotypes)
    
    @property
    def graph(self):
        data_set = self.get_top_phenotypes
        return pyplot.plot(data_set[0], data_set[1])


class RecorderDouble(Recorder):
    """
    二目的最適化のレコーダー
    コンストラクタで上位nランクの個体群を記録
    """
    def __init__(self, record_num = 1):
        super.__init__(record_num)
        self.__tops_of_popurations = []
    
    def record(self, population:POP.Population):
        self.__tops_op_popurations.append(population.get_rank_population)
    
    def graph(self, generation, rank = 0):
        data_set = self.tops_op_popurations[generation][rank]
        data_x = [individual.phenotype[0] for individual in data_set]
        data_y = [individual.phenotype[1] for individual in data_set]
        return pyplot.plot(data_x, data_y)



def recorder_builder(objective_num = 1):
    """
    Recorderを生成する関数
    単一目的最適化ならRecorderSingleを
    2目的ならRecordDoubleを
    多目的ならRecordMultiを返す
    """
    if objective_num == 1:
        return Recorder
    else:
        return RecorderDouble
        
        