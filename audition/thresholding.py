from audition.utils import str_in_sql
import logging


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

    def model_groups_close_to_best_as_of(self, train_end_time):
        query = ' intersect '.join(['''
            select model_group_id
            from {dist_table} dist
            where
            train_end_time = '{train_end_time}'
            and model_group_id in ({model_group_ids})
            and metric = '{metric}'
            and parameter = '{parameter}'
            and below_best < {max_below}
        '''.format(
            dist_table=self.distance_from_best_table.distance_table,
            model_group_ids=str_in_sql(self.model_group_ids),
            train_end_time=train_end_time,
            metric=metric_filter['metric'],
            parameter=metric_filter['metric_param'],
            max_below=metric_filter['max_below_best'],
        ) for metric_filter in self._metric_filters])
        return set(row[0] for row in self.db_engine.execute(query))

    def model_groups_above_min_as_of(self, train_end_time):
        query = ' intersect '.join(['''
            select model_group_id
            from {dist_table} dist
            where
            train_end_time = '{train_end_time}'
            and model_group_id in ({model_group_ids})
            and metric = '{metric}'
            and parameter = '{parameter}'
            and raw_value >= {min_value}
        '''.format(
            dist_table=self.distance_from_best_table.distance_table,
            model_group_ids=str_in_sql(self.model_group_ids),
            train_end_time=train_end_time,
            metric=metric_filter['metric'],
            parameter=metric_filter['metric_param'],
            min_value=metric_filter['min_value'],
        ) for metric_filter in self._metric_filters])
        return set(row[0] for row in self.db_engine.execute(query))

    def model_groups_passing_rules(self):
        below_min_model_groups = set(self._model_group_ids)
        close_to_best_model_groups = set()
        for train_end_time in self.train_end_times:
            close_to_best = self.model_groups_close_to_best_as_of(train_end_time)
            logging.warning(
                'Found %s model groups close to best for %s',
                len(close_to_best),
                train_end_time
            )
            close_to_best_model_groups |= close_to_best

            below_min = self.model_groups_above_min_as_of(train_end_time)
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
