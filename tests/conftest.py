import os

import pandas as pd
import pytest


@pytest.fixture
def optimize_warning_bar_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'bar_optimization/data/optimize_warning_bar_data.csv')
    return pd.read_csv(data_path, parse_dates=['time'])
