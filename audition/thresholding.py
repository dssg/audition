import logging


def _above_min(df, metric_filter):
    return df[df['raw_value'] >= metric_filter['min_value']]


def _close_to_best(df, metric_filter):
    return df[df['below_best'] < metric_filter['max_below_best']]


def _of_metric(df, metric_filter):
    return df[
        (df['metric'] == metric_filter['metric']) &
        (df['parameter'] == metric_filter['metric_param'])
    ]


class ModelGroupThresholder(object):
    def __init__(
        self,
        db_engine,
        distance_from_best_table,
        train_end_times,
        initial_model_group_ids,
    ):
        self.db_engine = db_engine
        self.distance_from_best_table = distance_from_best_table
        self.train_end_times = train_end_times
        self._model_group_ids = initial_model_group_ids
        self._metric_filters = []

    def _filter_model_groups(self, df, filter_func):
        passing = set(self._model_group_ids)
        for metric_filter in self._metric_filters:
            passing &= set(filter_func(
                _of_metric(df, metric_filter),
                metric_filter
            )['model_group_id'])
        return passing

    def model_groups_above_min(self, df):
        return self._filter_model_groups(df, _above_min)

    def model_groups_close_to_best(self, df):
        return self._filter_model_groups(df, _close_to_best)

    def model_groups_passing_rules(self):
        below_min_model_groups = set(self._model_group_ids)
        close_to_best_model_groups = set()
        for train_end_time in self.train_end_times:
            df_as_of = self.distance_from_best_table.dataframe_as_of(
                model_group_ids=self._model_group_ids,
                train_end_time=train_end_time,
            )
            close_to_best = self.model_groups_close_to_best(df_as_of)
            logging.warning(
                'Found %s model groups close to best for %s',
                len(close_to_best),
                train_end_time
            )
            close_to_best_model_groups |= close_to_best

            below_min = self.model_groups_above_min(df_as_of)
            logging.warning(
                'Found %s model groups below min for %s',
                len(below_min),
                train_end_time
            )
            below_min_model_groups &= below_min

        total_model_groups = close_to_best_model_groups & below_min_model_groups
        logging.warning(
            'Found %s total model groups past threshold',
            len(total_model_groups)
        )
        return total_model_groups

    def update_filters(self, new_metric_filters):
        if new_metric_filters != self._metric_filters:
            self._metric_filters = new_metric_filters
            self._model_group_ids = self.model_groups_passing_rules()

    @property
    def model_group_ids(self):
        return self._model_group_ids
