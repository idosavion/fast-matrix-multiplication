from subprocess import Popen, PIPE

import ResultParser

from typing import List
from ParseStrassenVSMkl import parse_current_log


class RunTimeComparator:
    def __init__(self, command: List[str], iter_num: int, parser: ResultParser):
        self._parser = parser
        self._command = command
        self._iter_num = iter_num

    def create_comparison(self):
        p = Popen(self._command, stdout=PIPE)
        lines = []
        all_results = []
        for i in range(self._iter_num):
            current_execution_results = self._get_execution_result(lines, p)
            all_results += current_execution_results


    def _get_execution_result(self, lines: List[str], p: object) -> List[ResultParser.Result]:
        for line in iter(p.stdout.readline, ""):
            lines.append(line)
        results = self._parser.get_results()
        return results
