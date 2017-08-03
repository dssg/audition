import copy
from audition.selection_rules import *
import numpy
import pandas
from audition.plotting import plot_cats


class BoundSelectionRule(object):
    def __init__(self, descriptive_name, function, args):
        self.descriptive_name = descriptive_name
        self.function = function
        self.args = args

    def pick(self, dataframe, train_end_time):
        return self.function(dataframe, train_end_time, **(self.args))


class RegretCalculator(object):
    def __init__(self, distance_from_best_table):
        """Calculates 'regrets' for different model group selection rules

        A regret refers to the difference in performance between a model group
        and the best model group for the next testing window
        if a selection rule is followed.

        Args:
            distance_from_best_table (audition.DistanceFromBestTable)
                A pre-populated distance-from-best database table
        """
        self.distance_from_best_table = distance_from_best_table

    def regrets_for_rule(
        self,
        bound_selection_rule,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter,
    ):
        """Calculate the regrets, or distance between the chosen model and
            the maximum value next test time

        Arguments:
            bound_selection_rule (audition.regrets.BoundSelectionRule) A function that returns a model group
                given a dataframe of model group performances plus other
                arguments
            model_group_ids (list) The list of model group ids to include in
                the regret analysis
            train_end_times (list) The list of train end times to include in
                the regret analysis
            metric (string) -- model evaluation metric, such as 'precision@'
            parameter (string) -- model evaluation metric parameter,
                such as '300_abs'

        Returns: (list) for each train end time, the distance between the
            model group chosen by the selection rule and the potential
            maximum for the next train end time
        """
        regrets = []
        df = self.distance_from_best_table.as_dataframe(model_group_ids)

        for train_end_time in train_end_times:
            localized_df = copy.deepcopy(
                df[df['train_end_time'] <= train_end_time]
            )
            del localized_df['dist_from_best_case_next_time']

            choice = bound_selection_rule.pick(localized_df, train_end_time)
            regret_result = df[
                (df['model_group_id'] == choice) &
                (df['train_end_time'] == train_end_time) &
                (df['metric'] == regret_metric) &
                (df['parameter'] == regret_parameter)
            ]
            assert len(regret_result) == 1
            regrets.append(regret_result['dist_from_best_case_next_time'].values[0])
        return regrets


class SelectionRulePlotter(object):
    def __init__(self, regret_calculator):
        self.regret_calculator = regret_calculator

    def create_plot_dataframe(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter
    ):
        accumulator = list()
        for selection_rule in bound_selection_rules:
            regrets = self.regret_calculator.regrets_for_rule(
                selection_rule,
                model_group_ids,
                train_end_times,
                regret_metric,
                regret_parameter
            )
            for pct_diff in range(0, 100):
                pct_of_time = numpy.mean([1 if regret < pct_diff else 0 for regret in regrets])
                accumulator.append({
                    'pct_diff': pct_diff,
                    'pct_of_time': pct_of_time,
                    'selection_rule': selection_rule.descriptive_name,
                })
        return pandas.DataFrame.from_records(accumulator)

    def plot_all_selection_rules(
        self,
        bound_selection_rules,
        model_group_ids,
        train_end_times,
        regret_metric,
        regret_parameter
    ):
        df_regrets = self.create_plot_dataframe(
            bound_selection_rules,
            model_group_ids,
            train_end_times,
            regret_metric,
            regret_parameter
        )
        cat_col = 'selection_rule'
        plt_title = 'Fraction of models X pp worse than best {} {} next time'.format(regret_metric, regret_parameter)

        plot_cats(
            frame=df_regrets,
            x_col='pct_diff',
            y_col='pct_of_time',
            cat_col=cat_col,
            title=plt_title,
            x_label='decrease in {} next time from best model'.format(regret_metric),
            y_label='fraction of models',
            x_ticks=numpy.arange(0, 1.1, 0.1)
        )
