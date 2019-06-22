from collections import namedtuple


Result = namedtuple('Result', ['dim', 'algorithm', 'time'])


class ResultParser:
    def __init__(self, file_path: str, parse_func):
        with open(file_path, 'r') as results:
            lines = results.readlines()
            self._results = parse_func(lines)

    def print_results(self):
        for result in self._results:
            print(result)

    def get_results(self):
        return self._results



