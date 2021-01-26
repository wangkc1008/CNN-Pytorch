"""
created by PyCharm
date: 2021/1/15
time: 0:16
user: wkc
"""
from collections import namedtuple
from itertools import product


class RunBuilder:
    @staticmethod
    def get_run(params):  # 静态方法，不需要实例化

        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
