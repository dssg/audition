from catwalk.evaluation import ModelEvaluator
import logging


def str_in_sql(values):
    return ','.join(map(lambda x: "'{}'".format(x), values))


def greater_is_better(metric):
    if metric in ModelEvaluator.available_metrics:
        return ModelEvaluator.available_metrics[metric].greater_is_better
    else:
        logging.warning(
            'Metric %s not found in available metrics, assuming greater is better',
            metric
        )
        return True
