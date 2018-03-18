#coding: UTF-8
import individual as INDI
import utilities as UTIL
from module.algos import select as SEL
from module.algos import crossover as CRV
from module.algos import mutation as MUT
from typing import Optional
from typing import List
from numba import jit 


class GA(object):
    """
    遺伝的アルゴリズムを流す本体
    初期化する際に各種アルゴリズムに加え、個体を生成する（遺伝子に対して適用する問題をラップしたもの）
    を代入する必要あり
    """
    def __init__(self, problem, **config):
        """
        ここにアルゴリズムなどを入れて初期化
        初期化する変数の詳細
        is_max:bool 最大化問題かどうか　デフォルトではTrue
        is_multi_objective:bool 多目的最適かどうか判定　デフォルトではFalse
        select:SEL.Select 選択操作を行うオブジェクトが入る　デフォルトでは2個体をトーナメント
        crossover:CRV.Crossover 交差操作を行うオブジェクトが入る　デフォルトでは一様交差，交叉確率1.0
        mutation:MUT.Mutation 突然変異操作を表すオブジェクトが入る　デフォルトではNone(遺伝子の型によって挙動が大きく異なるため)
        """
        if "is_multi_objective" not in config:
            is_multi_objective = False
        else:
            is_multi_objective = config["is_multi_objective"]
        
        if "is_max" not in config:
            is_max = True
        else:
            is_max = config["is_max"]            
        self.__incubator = INDI.incubator_builder(problem, is_max, is_multi_objective)
        if "select" not in config:
            self.__select = SEL.DEFAULT
        else:
            self.__select = config["select"]
        if "crossover" not in config:
            self.__crossover = None
        else:
            self.__crossover = config["crossover"]
        if "mutation" not in config:
            self.__mutation = MUT.DEFAULT
        else:
            self.__mutation = config["mutation"]
    
    @property
    def select(self):
        return self.__select
    
    @property
    def crossover(self):
        return self.__crossover
    
    @property
    def mutation(self):
        return self.__mutation
    
    @property
    def incubator(self):
        return self.__incubator
    
    def incubate_genom_set(self, genom_set:List[INDI.Locus]):
        """
        作成された遺伝子のセットから個体群を作成する
        """    
        return INDI.Populaion([self.incubator(genom) for genom in genom_set])
    
    @jit
    def build_children(self, base_pop, **config) ->INDI.Population:
        """
        個体の掛け合わせを行う
        引数の一覧は以下の通り
        base_pop:INDI.population ベースとなる個体群
        pairents_num:int 掛け合わせに必要となる親の数　基本は2だがアルゴリズムによっては複数個体を親にすることもあるので
        prob_crossover:double 交叉確率　交叉オブジェクトを初期化する際にも設定しているが，動的に確立を変更させたいケースもあるので デフォルトではNone
        prob_mutation:double 突然変異確率　交叉確率と同文        
        """
        if "pairents_num" not in config:
            pairents_num = 2
        else:
            pairents_num = config["pairents_num"]
        if "prob_crossover" not in config:
            prob_crossover = None
        else:
            prob_crossover = config["prob_crossover"]
        if "prob_mutation" not in config:
            prob_mutation = None
        else:
            prob_mutation = config["prob_mutation"]
        parents = self.select(base_pop, pairents_num)
        crossed_genom_set = self.crossover(*parents, prob_crossover)
        if self.mutation is None:
            mutated_genom_set = crossed_genom_set
        else:
            mutated_genom_set = self.mutation(crossed_genom_set)
        return self.incubate_genom_set(mutated_genom_set)
    
    @jit
    def run_one_generation(self, base_pop, surviver_percent:int = 50, pairents_num:Optional[int] = None, prob_crossover:float = 1.0, prob_mutation:float = 0.01) ->INDI.Population:
        
        children = self.build_children(base_pop, pairents_num, prob_crossover, prob_mutation)
        return base_pop.merge(children).elite(surviver_percent)
    
    @jit
    def init_params(self, generation_num:int, first_pop:INDI.Population):
        def fit(surviver_percent:int = 50, pairents_num:int = 2, prob_crossover:float = 1.0, prob_mutation:float = 0.01, record_num:int = 1):
            recorder = UTIL.Recorder(record_num)
            pool = first_pop[:]
            for generation in range(generation_num):
                recorder.record(pool)
                pool = self.run_one_generation(pool[:], surviver_percent ,pairents_num, prob_crossover, prob_mutation)
            return recorder
        
        return fit
