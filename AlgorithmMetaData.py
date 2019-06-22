import math


class AlgorithmMetaData:
    def __init__(self, n: int, m: int, k: int, t: int,name="",run_time_func=None):
        self.n = n
        self.m = m
        self.k = k
        self.t = t
        self.w_0 = self._calculate_asymptotic_complexity()
        self._set_run_time_func(run_time_func)
        self.name = name

    def _set_run_time_func(self, run_time_func):
        if run_time_func is None:
            self.run_time_func = lambda x: x ** self.w_0
        else:
            self.run_time_func = run_time_func

    def _calculate_asymptotic_complexity(self) -> float:
        return math.log(self.t ** 3, self.n * self.m * self.k)

    def calculate_runtime(self, dim):
        return self.run_time_func(dim)
