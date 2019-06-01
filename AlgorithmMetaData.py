import math


class AlgorithmMetaData:
    def __init__(self, n: int, m: int, k: int, t: int,name=""):
        self.n = n
        self.m = m
        self.k = k
        self.t = t
        self.w_0 = self._calculate_asymptotic_complexity()
        self.run_time_func = lambda x: x ** self.w_0
        self.name = name

    def _calculate_asymptotic_complexity(self) -> float:
        return math.log(self.t ** 3, self.n * self.m * self.k)

    def calculate_runtime(self, dim):
        return self.run_time_func(dim)
