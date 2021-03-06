from audition.model_group_performance import ModelGroupPerformancePlotter
import testing.postgresql
from sqlalchemy import create_engine
import numpy
from tests.utils import create_sample_distance_table
from unittest.mock import patch


def test_ModelGroupPerformancePlotter_generate_plot_data():
    with testing.postgresql.Postgresql() as postgresql:
        engine = create_engine(postgresql.url())
        distance_table, model_groups = create_sample_distance_table(engine)
        plotter = ModelGroupPerformancePlotter(distance_table)
        df = plotter.generate_plot_data(
            metric='precision@',
            parameter='100_abs',
            model_group_ids=[1, 2],
            train_end_times=['2014-01-01', '2015-01-01'],
        )
        assert sorted(df['model_type'].unique()) == ['best case', 'mySpikeClassifier', 'myStableClassifier']
        for value in df[df['model_group_id'] == 1]['raw_value'].values:
            assert numpy.isclose(value, 0.5)


def test_ModelGroupPerformancePlotter_plot_all():
    with patch('audition.model_group_performance.plot_cats') as plot_patch:
        with testing.postgresql.Postgresql() as postgresql:
            engine = create_engine(postgresql.url())
            distance_table, model_groups = create_sample_distance_table(engine)
            plotter = ModelGroupPerformancePlotter(distance_table)
            plotter.plot_all(
                [{'metric': 'precision@', 'parameter': '100_abs'}],
                model_group_ids=[1, 2],
                train_end_times=['2014-01-01', '2015-01-01'],
            )
        assert plot_patch.called
        args, kwargs = plot_patch.call_args
        assert 'raw_value' in kwargs['frame']
        assert 'train_end_time' in kwargs['frame']
        assert kwargs['x_col'] == 'train_end_time'
        assert kwargs['y_col'] == 'raw_value'
