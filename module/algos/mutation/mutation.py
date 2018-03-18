#こちらは突然変異のアルゴリズムを記述
#返し値は遺伝子の配列
#ここにあるアルゴリズムを引数にして使う
import numpy as np
import utilities as UTIL
from typing import Union
from typing import Optional
from typing import Callable
from typing import TypeVar
from typing import List
import individual as INDI

#型ヒントエイリアス
T = TypeVar('T')
MutationFunc = Callable[Union[INDI.individualBase, list[T], np.ndarray[T]], Union[List[T], np.ndarray]]
MutationFuncList = Callable[Union[INDI.individualBase, list[T], np.ndarray[T]], List[T]]
MutationFuncNp = Callable[Union[INDI.individualBase, list[T], np.ndarray[T]], np.ndarray]

#確率に応じて変更するかどうか判定
def is_change(genom_length: int,  prob: float) -> np.ndarray :
    rand_list = np.random.rand(genom_length)
    return np.array([ r < prob for r in rand_list])

#変更する場所のインデックスを取得
def search_change_index(will_change_list) -> List[bool]:
    return [index for index, will_change in enumerate(will_change_list)
               if will_change is True]

#バイナリ記号列向け　ビット反転
def inverse_bit(individual, prob = 0.01) -> List[int]:
    mask_list = is_change(len(individual[:]), prob)
    inverse = lambda geno: 1 if geno == 0 else 0
    new_gene = []
    for geno, mask in zip(individual, mask_list):
        if mask is False:
            new_gene.append(inverse(geno))
        else:
            new_gene.append(geno)
    return new_gene

#順列向け　隣の配列を入れ替え
def exchange_adjacent(individual:Union[List[int], INDI.IndividualBase], prob: float = 0.01 ) -> List[int]:
    change_list = is_change(len(individual[:])-1, prob)    
    new_gene = individual[:]
    for change_point in change_list:
        new_gene[change_point], new_gene[change_point+1] = new_gene[change_point+1], new_gene[change_point]
    return new_gene

#実数値向け　一様乱数付加
def randomize(max_list:Optional[np.ndarray] = None, min_list:Optional[np.ndarray] = None) -> np.ndarray:
    max_vec, min_vec, range_list =  UTIL.set_max_and_min(max_list, min_list)
    
    def get_change_vector(indi_num, prob):
        change_list = is_change(indi_num, prob)
        keep_list = np.array([not(attribute) for attribute in change_list])
        return (keep_list, get_mutated_index(change_list))

    def get_mutated_index(changed_list, ranges):
        give_rand = lambda will_change: np.random.rand() if will_change is True else 0
        return np.array([give_rand(attribute) for attribute in changed_list])

    def mutation(individual, prob = 0.01):
        mask_list = get_change_vector(len(individual[:]), prob)
        new_gene = individual*mask_list[0] + range_list*mask_list[1]
        return  UTIL.cut_upper_and_under(new_gene, max_list, min_list)

    return mutation

#実数値向け　正規乱数を足す
def add_gaussian_rand(max_list:Optional[np.ndarray] = None, min_list:Optional[np.ndarray] = None, variance = 1.0) -> np.ndarray:
    max_vec, min_vec, range_list =  UTIL.set_max_and_min(max_list, min_list) 

    def get_mutated_index(change_list):
         give_r_gaussian = lambda will_change:np.random.randn(0, variance) if will_change is True else 0
         return np.array([give_r_gaussian(index) for index in change_list])

    def mutation(individual, prob = 0.01):
         mask = is_change(len(individual[:]), prob)
         add_list = get_mutated_index(mask)
         new_gene = individual[:] + add_list*range_list
         return UTIL.cut_upper_and_under(new_gene, max_list, min_list)
     
    return mutation

def get_mutation_of_binary(mutation_type:str = "INV") -> MutationFuncList:
    index_to_func = {}
    index_to_func["INV"] = inverse_bit
    if mutation_type in index_to_func:
        return index_to_func[mutation_type]
    return inverse_bit


def get_mutation_of_order(mutation_type:str = "EXC") -> MutationFuncList:
    index_to_func = {}
    index_to_func["EXC"] = exchange_adjacent
    if mutation_type in index_to_func:
        return index_to_func[mutation_type]
    return exchange_adjacent


def get_mutation_of_real(mutation_type:str = "RAND") -> MutationFuncList:
    index_to_func = {}
    index_to_func["RAND"] = randomize
    index_to_func["GAU"] = add_gaussian_rand
    if mutation_type in index_to_func:
        return index_to_func[mutation_type]
    return randomize    
 
#個体の設定からアルゴリズムを呼び出す
def get_algo_from_genotype(conf_genotype:dict) -> MutationFunc:
    genotype_to_algo = {}
    genotype_to_algo["BIN"] = get_mutation_of_binary
    genotype_to_algo["ORD"] = get_mutation_of_order
    genotype_to_algo["REAL"] = get_mutation_of_real
    if conf_genotype not in genotype_to_algo:
        return None
    else:
        return genotype_to_algo[conf_genotype]    


#突然変異のクラス。上記の関数をコンストラクタに代入して実行
class Mutation(object):
    def __init__(self, mutation_algo, prob = 0.01):
        self.__algo = mutation_algo
        self.__prob_default = prob
    
    def execute(self, individual:Union[List[T], INDI.Locus], prob:Optional[float] = None) -> INDI.Locus:
        if prob is None:
            mutate_prob = self.__prob_default
        else:
            mutate_prob = prob
        return self.__algo(individual, mutate_prob)
    
    def execute_group(self, population:Union[List[INDI.BaseIndividual], INDI.Population], prob:Optional[float] = None) -> INDI.Locus:
        return [self.execute(individual, prob) for individual in population]

