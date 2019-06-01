from Algorithms import *
from AlgorithmMetaData import AlgorithmMetaData
from typing import List
import matplotlib.pyplot as plt


class AlgorithmsComparator:
    def __init__(self, algorithms: List[AlgorithmMetaData]):
        self.algorithms = algorithms

    def compare(self, dims: List[int]):
        for algorithm in self.algorithms:
            run_time = [algorithm.calculate_runtime(dim) for dim in dims]
            plt.plot(dims, run_time)
        plt.show()


if __name__ == '__main__':
    ac = AlgorithmsComparator([classical, original_algorithm, alternative_basis, decomposed, fully_decomposed])
    ac.compare(list(range(1,200)))