#coding: UTF-8
import numpy as np
import random 
from ...util import utilities as UTIL
from ...indi.individual import BaseIndividual
from ...indi.individual import Locus
from typing import TypeVar
from typing import Union
from typing import Optional
from typing import List
from typing import Tuple
from numba import jit 
#こちらは交叉のアルゴリズムを記述
#返し値は遺伝子の配列
#ここにあるアルゴリズムを引数にして使う

T = TypeVar('T')
Parent = Union[BaseIndividual, Locus]

#基本アルゴリズムは動作検証済み

#基本的には交差自体の操作そのものは親2つで十分だけど一部の実装だと複数の親を選ばなきゃ
#ならない。インターフェースの統一も兼ねて可変長引数にしてるけど基本的には第3引数意向は無視されるよ


@jit
def one_point_crossover(*parents:Tuple[Parent,Parent]) -> List[T]:
    """
    一点交叉
    インプットは個体2つだけどインターフェース統一のために任意の数だけ親を撮れるように
    三番目以降は無視されるよ
    """
    father = parents[0]#あなたがパパになるんですよ
    mother = parents[1]#お前がママになるんだよ！
    point = np.random.randint(0, len(father[:])) 
    print(point)
    if  random.randint(0, 2) == 0:
        gene = father[:point]
        gene.extend(mother[point:])
    else:
        gene = mother[:point]
        gene.extend(father[point:])
    return gene

@jit    
def two_point_crossover(*parents:Tuple[Parent,Parent]) -> List[T]:
    """
    二点交差
    インプットは個体2つだけどインターフェース統一のために任意の数だけ親を撮れるように
    三番目以降は無視されるよ
    """
    father = parents[0]#あなたがパパになるんですよ
    mother = parents[1]#お前がママになるんだよ！
    point1 = random.randint(0, len(father[:]))
    point2 = random.randint(point1, len(father[:]))
    if  np.random.randint(0, 2) == 0:
        gene = father[:point1]
        gene.extend(mother[point1:point2])
        gene.extend(father[point2:])
    else:
        gene = mother[:point1]
        gene.extend(father[point1:point2])
        gene.extend(mother[point2:])
    return gene


def uniformity_crossover(*parents:Tuple[Parent,Parent]) -> List[T]:
    """
    一様交叉
    インプットは個体2つだけどインターフェース統一のために任意の数だけ親を撮れるように
    三番目以降は無視されるよ
    """    
    father = parents[0]#あなたがパパになるんですよ
    mother = parents[1]#お前がママになるんだよ！
    mask_list = [np.random.randint(0, 2) for x in range(len(father[:]))]
    select_gene = lambda geno1, geno2, mask: geno1 if mask == 0 else geno2
    return [select_gene(genom1, genom2, mask) for genom1, genom2, mask in zip(father, mother, mask_list)]



def replacement(replacement_num:int = 2):
    """
    順列を入れ替える。巡回セールスマン問題など遺伝子の並びが意味を成す場合に使う
    こちらは入れ替える数を初期化
    返し値はアルゴリズムを実行する関数
    """
    def crossover(*parents:Tuple[Parent,Parent]) -> List[T]:
        """
        アルゴリズム本体
        インプットは個体2つだけどインターフェース統一のために任意の数だけ親を撮れるように
        三番目以降は無視されるよ
        """
        father = parents[0]#あなたがパパになるんですよ
        mother = parents[1]#お前がママになるんだよ！
        def sampling (base_gene, sampled = None):
            if sampled is None:
                sampled_index = random.sample(father[:], replacement_num)
            else:
                sampled_index = sampled
            return [genom for genom in base_gene if genom in sampled_index]
        
        def replace(base_gene, dict_index):
            converter = lambda genom : dict_index[genom] if genom in dict_index else genom
            return [converter(genom) for genom in base_gene]            
        
        father_order = sampling(father)
        mother_order = sampling(mother, father_order)
        index_father_to_mother = {father_index: mother_index
                                  for father_index, mother_index in zip(father_order, mother_order)}
        index_mother_to_father = {mother_index: father_index
                                  for father_index, mother_index in zip(father_order, mother_order)}                
        children = [replace(father,  index_father_to_mother), replace(mother, index_mother_to_father)]
        return random.choice(children)
    return crossover
        

#ここからは実数型。範囲は0<genom<1 範囲を持つ場合は問題側で対応する
def blx_alpha(min_list = None, max_list = None, alpha = 0.2):
    """
    BLX-アルファ　遺伝子が実数の場合に使う。
    はじめにアルファを設定
    返し値はアルゴリズムを実行する関数
    """    
    def cross_over(*parents:Tuple[Parent,Parent]) -> np.ndarray:
        """
        アルゴリズムを実行する関数
        インプットは個体2つだけどインターフェース統一のために任意の数だけ親を撮れるように
        三番目以降は無視されるよ
        """
        father = parents[0]#あなたがパパになるんですよ
        mother = parents[1]#お前がママになるんだよ！        
        geno1 = np.array(father[:])
        geno2 = np.array(mother[:])
        mid = (geno1+geno2)/2.0
        bases = np.abs(geno1-mid)*(1.0+alpha)
        #乱数の範囲は-1.0から1.0の実数
        array_rand = 2.0*np.random.rand(len(bases))-np.array([1.0 for x in range(len(bases))])
        return UTIL.cut_upper_and_under(mid+bases*array_rand, max_list, min_list)
    return cross_over


#ここからは関数の呼び出し型を定義

def get_crossover_of_binary(crossover_type:str = "UNI"):
    """
    バイナリ記号列を遺伝子型とする場合の交叉のアルゴリズムを呼び出す
    ONEなら一点交差
    TWOなら二点交差
    UNIなら一様交差
    のアルゴリズムを流す関数を返す
    """
    index_to_func = {}
    index_to_func["ONE"] = one_point_crossover
    index_to_func["TWO"] = two_point_crossover
    index_to_func["UNI"] = uniformity_crossover
    if crossover_type in index_to_func:
        return index_to_func[crossover_type]
    else:
        return uniformity_crossover


#順列を遺伝子型とする場合の交叉のアルゴリズムを呼び出す
def get_crossover_of_order(crossover_type = "REP"):
    index_to_func = {}
    index_to_func["REP"] = replacement
    if crossover_type in index_to_func:
        return index_to_func[crossover_type]
    else:
        return replacement


#実数の配列を遺伝子型とする場合の交叉のアルゴリズムを呼び出す
def get_crossover_of_real(crossover_type:str = "BLX"):
    """
    遺伝子型が実数の場合の考査のアルゴリズムを返す
    2018年3月12日現在BLX-アルファしか実装できていないのでそれしか返さない
    """
    index_to_func = {}
    index_to_func["BLX"] = blx_alpha
    if crossover_type in index_to_func:
        return index_to_func[crossover_type]
    else:
        blx_alpha
    

#個体の設定からアルゴリズムを呼び出す
def get_algo_from_genotype(conf_genotype:str):
    """
    設定ファイルに書かれた遺伝子型に応じてどの交差を呼び出すか
    判定する
    """
    genotype_to_algo = {}
    genotype_to_algo["BIN"] = get_crossover_of_binary
    genotype_to_algo["ORD"] = get_crossover_of_order
    genotype_to_algo["REAL"] = get_crossover_of_real
    if conf_genotype not in genotype_to_algo:
        return None
    else:
        return genotype_to_algo[conf_genotype]


#setting.jsonから読み込んだdict型の設定から使用する関数を呼び出す
def call_func_from_conf(conf):
    """
    設定ファイルから使用する関数を呼び出す
    """
    genotype = conf["individual"]["genotype"]
    setting = conf["clossover"]
    func_list = get_algo_from_genotype(genotype)
    if genotype == "BIN":
        use_func = func_list(setting["type"])
    if genotype == "ORD":
        use_func_base = func_list(setting["type"])
        if setting["type"] == "REP":
            use_func = use_func_base(setting["type"]["replacement_num"])
    if genotype == "REAL":
        geno_length = conf["individual"]["gene_length"]
        max_list, min_list = UTIL.build_max_min_list(geno_length, setting["max"], setting["min"])
        use_func_base = func_list(setting["type"])
        if setting["type"] == "BLX":
            use_func = use_func_base(setting["alpha"], max_list, min_list)
    return Crossover(use_func, setting["prob"])


class Crossover(object):
    """
    交差に関してのセッティングを記述するクラス
    コンストラクタで使用するアルゴリズムと交差確立を設定する。
    交差アルゴリズムはここで記述されているもの以外に自分で
    作成したアルゴリズムを入れ込むことも可能
    """
    def __init__(self, crossover_algo, prob = 1.0):
        self.__algo = crossover_algo
        self.__prob_default = prob
        self.__rand = np.random.rand
    
    @jit
    def execute(self, *parents:Parent, prob:Optional[float] = None) -> Optional[Locus]:
        if prob is None:
            cross_prob = self.__prob_default
        else:
            cross_prob = prob
        r = self.__rand()
        if r < cross_prob:
            return self.__algo(*parents)
        return None
    
    def execute_group(self, *parents:Parent, prob:Optional[float] = None) -> List[Locus]:
        return [self.execute(parent, prob) for parent in parents
                if self.execute(parent, prob) is not None]
        
        
