from catwalk.evaluation import ModelEvaluator
import logging
from collections import namedtuple


def str_in_sql(values):
    return ','.join(map(lambda x: "'{}'".format(x), values))



BoundMetric = namedtuple('BoundMetric', ['metric', 'parameter'])
