import math

from AlgorithmMetaData import AlgorithmMetaData

classical = AlgorithmMetaData(3, 3, 3, 27)
classical.run_time_func = lambda x: 2 * (x ** 3) - x ** 2

original_algorithm = AlgorithmMetaData(3, 3, 3, 23)
original_algorithm.run_time_func = lambda x: 7.93 * (x ** math.log(23, 3)) - 6.93 * (x ** 2)

alternative_basis = AlgorithmMetaData(3, 3, 3, 23)
alternative_basis.run_time_func = lambda x: 5.36 * (x ** math.log(23, 3)) + 3.22 * (x ** 2) * math.log(x, 3) - 4.36 * (
        x ** 2)

decomposed = AlgorithmMetaData(3, 3, 3, 23)
decomposed.run_time_func = lambda x: 2 * (x ** math.log(23, 3)) + 6.75 * (x ** math.log(3, 21)) - 7.75 * (x ** 2)

fully_decomposed = AlgorithmMetaData(3, 3, 3, 23)
fully_decomposed.run_time_func = lambda x: 2 * (x ** math.log(23, 3)) + 3 * (x ** math.log(20, 3)) + \
                                           2 * (x ** math.log(14, 3) + 2 * (x ** math.log(12, 3)) +
                                                2 * (x ** math.log(11, 3) + 33 * (x ** math.log(10, 3)) -
                                                     43 * (x ** 2)))
