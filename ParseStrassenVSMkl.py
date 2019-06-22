import os
import re
import sys
from itertools import groupby
from typing import List

from ResultParser import Result, ResultParser

ANY_NON_DIGIT_OR_FLOATING_POINT = r"[^0-9.]"


def _split_lines_by_using_dimension_line(lines: List[str]) -> List[List[str]]:
    separating_lines_indices: List[int] = [i for i, line in enumerate(lines) if "Using dim" in line] + [len(lines)]
    splitted_lines: List[List[str]] = []
    number_of_sections = range(len(separating_lines_indices) - 1)
    if number_of_sections == 0:
        return [lines]
    for i in number_of_sections:
        section = lines[separating_lines_indices[i]:separating_lines_indices[i + 1]]
        splitted_lines.append(section)
    return splitted_lines


def _parse_section(section: List[str]) -> List[Result]:
    current_dim: int = int(re.sub(r"[\D]", "", section[0]))
    results = []
    for line in section[1:]:
        result = _build_result(current_dim, line)
        results.append(result)
    return results


def _build_result(current_dim, line):
    algorithm, run_time = line.split('took', 1)
    run_time: float = float(re.sub(ANY_NON_DIGIT_OR_FLOATING_POINT, "", run_time))
    result = Result(current_dim, algorithm, run_time)
    return result


def parse_current_log(lines: List[str]):
    lines_splitted_by_matrix_dims: List[List[str]] = _split_lines_by_using_dimension_line(lines)
    results = []
    for section in lines_splitted_by_matrix_dims:
        results += _parse_section(section)
    return results


