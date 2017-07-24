from audition.utils import str_in_sql
from audition.plotting import plot_cats
import pandas as pd
import numpy as np


class DistanceFromBestTable(object):
    def __init__(self, db_engine, models_table, distance_table):
        self.db_engine = db_engine
        self.models_table = models_table
        self.distance_table = distance_table

    def _create(self):
        self.db_engine.execute('''create table {} (
            model_group_id int,
            model_id int,
            train_end_time timestamp,
            metric text,
            parameter text,
            raw_value float,
            below_best float,
            below_best_next_time float
        )'''.format(self.distance_table))

    def _populate(self, model_group_ids, train_end_times, metrics):
        for metric in metrics:
            self.db_engine.execute('''
                insert into {new_table}
                WITH model_ranks AS (
                    SELECT
                        m.model_group_id,
                        m.model_id,
                        m.train_end_time,
                        ev.value,
                        row_number() OVER (
                            PARTITION BY m.train_end_time
                            ORDER BY ev.value DESC, RANDOM()
                        ) AS rank
                  FROM results.evaluations ev
                  JOIN results.{models_table} m USING(model_id)
                  JOIN results.model_groups mg USING(model_group_id)
                  WHERE ev.metric='{metric}' AND ev.parameter='{metric_param}'
                        AND m.model_group_id IN ({model_group_ids})
                        AND train_end_time in ({train_end_times})
                ),
                model_tols AS (
                  SELECT train_end_time, model_group_id, model_id,
                         rank,
                         value,
                         first_value(value) over (
                            partition by train_end_time
                            order by rank ASC
                        ) AS best_val
                  FROM model_ranks
                ),
                current_best_vals as (
                    SELECT
                        model_group_id,
                        model_id,
                        train_end_time,
                        '{metric}',
                        '{metric_param}',
                        value as raw_value,
                        best_val - value below_best
                    FROM model_tols
                )
                select
                    current_best_vals.*,
                    first_value(below_best) over (
                        partition by model_group_id
                        order by train_end_time asc
                        rows between 1 following and unbounded following
                    ) below_best_next_time
                from current_best_vals
            '''.format(
                model_group_ids=str_in_sql(model_group_ids),
                train_end_times=str_in_sql(train_end_times),
                models_table=self.models_table,
                metric=metric['metric'],
                metric_param=metric['param'],
                new_table=self.distance_table
            ))

    def create_and_populate(
        self,
        model_group_ids,
        train_end_times,
        metrics,
    ):
        self._create()
        self._populate(model_group_ids, train_end_times, metrics)

    def as_dataframe(self, model_group_ids):
        return pd.read_sql(
            'select * from {} where model_group_id in ({})'.format(
                self.distance_table,
                str_in_sql(model_group_ids)
            ),
            self.db_engine
        )

    def dataframe_as_of(self, model_group_ids, train_end_time):
        base_df = self.as_dataframe(model_group_ids)
        return base_df[base_df['train_end_time'] == train_end_time]


class BestDistanceHistogrammer(object):
    def __init__(self, db_engine, distance_from_best_table):
        self.db_engine = db_engine
        self.distance_from_best_table = distance_from_best_table

    def generate_histogram_data(
        self,
        metric,
        metric_param,
        model_group_ids,
        train_end_times,
    ):
        """Fetch a best distance data frame from the distance table

        Arguments:
            metric (string) -- model evaluation metric, such as 'precision@'
            metric_param (string) -- model evaluation metric parameter,
                such as '300_abs'
        """
        model_group_union_sql = ' union all '.join([
            '(select {} as model_group_id)'.format(model_group_id)
            for model_group_id in model_group_ids
        ])
        sel_params = {
            'metric': metric,
            'metric_param': metric_param,
            'model_group_union_sql': model_group_union_sql,
            'distance_table': self.distance_from_best_table.distance_table,
            'model_group_str': str_in_sql(model_group_ids),
            'train_end_str': str_in_sql(train_end_times),
        }
        sel = """
                with model_group_ids as ({model_group_union_sql}),
                x_vals AS (
                  SELECT m.model_group_id, s.pct_diff
                  FROM
                  (
                  SELECT GENERATE_SERIES(0,100) / 100.0 AS pct_diff
                  ) s
                  CROSS JOIN
                  (
                  SELECT DISTINCT model_group_id FROM model_group_ids
                  ) m
                )
                SELECT dist.model_group_id, pct_diff, mg.model_type,
                       COUNT(*) AS num_models,
                       AVG(CASE WHEN below_best <= pct_diff THEN 1 ELSE 0 END) AS pct_of_time
                FROM {distance_table} dist
                JOIN x_vals USING(model_group_id)
                JOIN results.model_groups mg using (model_group_id)
                WHERE
                    dist.metric='{metric}'
                    AND dist.parameter='{metric_param}'
                    and model_group_id in ({model_group_str})
                    and train_end_time in ({train_end_str})
                GROUP BY 1,2,3
            """.format(**sel_params)

        return pd.read_sql(sel, self.db_engine)

    def plot_all_best_dist(self, metric_filters, model_group_ids, train_end_times):
        for metric_filter in metric_filters:
            df = self.generate_histogram_data(
                metric=metric_filter['metric'],
                metric_param=metric_filter['metric_param'],
                model_group_ids=model_group_ids,
                train_end_times=train_end_times
            )
            plot_best_dist(
                metric=metric_filter['metric'],
                metric_param=metric_filter['metric_param'],
                df_best_dist=df
            )


def plot_best_dist(metric, metric_param, df_best_dist, **plt_format_args):
    """Generates a plot of the percentage of time that a model group is
    within X percentage points of the best-performing model group using a
    given metric. At each point in time that a set of model groups is
    evaluated, the performance of the best model is calculated and the
    difference in performace for all other models found relative to this.

    An (x,y) point for a given model group on the plot generated by this
    method means that across all of those tets sets, the model
    from that model group performed within X percentage points of the best
    model in y% of the test sets.

    The plot will contain a line for each given model group representing
    the cumulative percent of the time that the group is within Xpp of the
    best group for each value of X between 0 and 100. All groups ultimately
    reach (1,1) on this graph (as every model group must be within 100pp of
    the best model 100% of the time), and a model specification that always
    dominated the others in the experiment would start at (0,1) and remain
    at y=1 across the graph.

    Arguments:
        metric (string) -- model evaluation metric, such as 'precision@'
        metric_param (string) -- model evaluation metric parameter, such as '300_abs'
        df_best_dist (pandas.DataFrame)
        **plt_format_args -- formatting arguments passed through to plot_cats()
    """

    cat_col = 'model_type'
    plt_title = 'Fraction of models X pp worse than best {} {}'.format(metric, metric_param)

    plot_cats(
        df_best_dist,
        'pct_diff',
        'pct_of_time',
        cat_col=cat_col,
        title=plt_title,
        x_label='decrease in {} from best model'.format(metric),
        y_label='fraction of models',
        x_ticks=np.arange(0, 1.1, 0.1),
        **plt_format_args
    )
