#coding: UTF-8
import numpy as np
from typing import Union
from typing import Optional
from typing import TypeVar
from typing import Callable
from typing import List
from typing import Generic

#型ヒントのエイリアス　Alias of type Hint
T = TypeVar('T') 
Locus = Union[List[T], np.ndarray]
Phenotype = Union[float, int, list, np.ndarray]
ProblemFunc = Optional[Callable[[Locus], Phenotype]]


class BaseIndividual(Generic[T]):
    def __init__(self, geno_base:Locus, is_max:bool, problem:ProblemFunc = None):
        self.__is_max = is_max
        self.__genotype = geno_base[:]
        if problem is None:
            print(problem)
        else:
            self.__phenotype = problem(self.genotype) 

    def __getitem__(self, geno_index) -> List[T]:
        return self.genotype[geno_index]

    def set_rank(self, population):
        """
        各個体のランクを設定する
        引数の説明
        population: Union[List[BaseIndividual], Population] 個体群
        適合度が同じであれば同じランク
        ランクの値が小さいほど優秀な個体
        所属する個体群に応じて算出するため、
        イミュータブルではなくなってしまう
        """
        self.__rank = 1
        for other in population:
            if self > other:
                self.__rank = self.__rank + 1
        

    @property
    def rank(self) -> Optional[int]:
        try:
            return self.__rank
        except AttributeError:
            return None    

    @property
    def genotype(self) -> Locus:
        return self.__genotype

    @property
    def is_max(self) -> bool:
        return self.__is_max

    @property
    def phenotype(self) -> Phenotype:
        return self.__phenotype

      

class Individual(BaseIndividual):
    """
    単一目的最適化での各個体
    """
    def __init__(self, geno_base:Locus, is_max:bool = True, problem:ProblemFunc = None):
        super().__init__(geno_base, is_max,  problem)

    def __lt__(self, other) -> bool:
        if self.is_max:
            return self.phenotype < other.phenotype
        else:
            return self.phenotype > other.phenotype


#各個体
class IndividualMultiObject(BaseIndividual):
    """
    多目的最適化での各個体
    """
    def __init__(self, geno_base:Locus, is_max:bool = True, problem:ProblemFunc = None):
        super().__init__(geno_base, problem)
        #numpy配列に
        self.__phenotype = np.array(self.__phenotype)

    def __lt__(self, other) -> bool:
        check_superior = self.phenotype - other.phenotype
        for attribute in check_superior:
            if self.is_max is True:
                if attribute > 0:
                    return False
            else:
                if attribute < 0:
                    return False
        return True
    


def incubator_builder(problem, is_max:bool, optimize_num:int = 1):
    """
    単一目的か多目的かに応じて個体を
    生成する個体のクラスを変更する
    """
    if optimize_num == 1:
        return lambda genom: IndividualMultiObject(genom, is_max, problem)
    else:
        return lambda genom: Individual(genom, is_max, problem)

