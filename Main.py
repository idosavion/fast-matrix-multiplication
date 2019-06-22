import os
import sys
from itertools import groupby
from typing import List, Any

from AlgorithmComparator import AlgorithmsComparator
from AlgorithmMetaData import AlgorithmMetaData
from ParseStrassenVSMkl import parse_current_log
from ResultParser import ResultParser, Result


def build_algorithm_from_logs():
    result: Result
    dim_to_time_dict = {dim: [] for dim in dims}
    for result in algorithm_results:
        dim_to_time_dict[result.dim].append(result.time)
    for key in dim_to_time_dict.keys():
        if dim_to_time_dict[key] == []:
            dim_to_time_dict[key] = 0
        else:
            dim_to_time_dict[key] = sum(dim_to_time_dict[key]) / len(dim_to_time_dict[key])
    dict_wrapper = lambda x: dim_to_time_dict.get(x)
    algorithm: AlgorithmMetaData = AlgorithmMetaData(3, 1, 1, 2, algorithm_name, dict_wrapper)
    return algorithm


def get_all_runs_results():
    list_of_files = os.listdir(path)
    logs = [file for file in list_of_files if file.startswith('log_')]
    algorithm_results: List[Result] = []
    for log in logs:
        result_parser = ResultParser(log, parse_current_log)
        algorithm_results += result_parser.get_results()
    return algorithm_results


if __name__ == '__main__':
    path = sys.argv[1]
    algorithm_results = get_all_runs_results()
    algorithm_results.sort(key=lambda x: x.algorithm)
    results_by_algorithm = groupby(algorithm_results, key=lambda x: x.algorithm)
    dims = sorted({result.dim for result in algorithm_results})
    algorithms: List[AlgorithmMetaData] = []
    for algorithm_name, algorithm_results in results_by_algorithm:
        algorithm = build_algorithm_from_logs()
        algorithms.append(algorithm)
    ac = AlgorithmsComparator(algorithms)
    ac.compare(dims)
