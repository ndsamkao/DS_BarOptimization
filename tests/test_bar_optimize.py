import pandas as pd

from bar_optimization.algo.optimize_warning_bar import OptimizeWarningBar


def test_optimize_bar(optimize_warning_bar_data):

    bar_model = OptimizeWarningBar(
        df=optimize_warning_bar_data,
        battery_capacity=15,
        battery_initial=0,
        contract_capacity=120,
        threshold=[0.9, 0.8, 0.7, 0.3],
        discharging_range=[7, 5, 3],  # kw
        charging_range=[5],  # kw
        charging_period=[0, 5],  # hour
    )

    result_df = bar_model.get_optimal_result(epoch=3).reset_index(drop=True)

    expected_df = pd.DataFrame(
        {
            'new_fees': [24196.804, 24196.804, 24196.804],
            'bar': [(0.83, 0.88, 0.91, 0.3), (0.81, 0.91, 0.98, 0.3), (0.81, 0.91, 0.95, 0.3)],
            'old_fees': [24196.804, 24196.804, 24196.804],
            'save': [0.0, 0.0, 0.0],
            'judge': [False, False, False],
        }
    )

    print(result_df[['save', 'judge']])
    print(expected_df[['save', 'judge']])
    assert result_df[['save', 'judge']].equals(expected_df[['save', 'judge']])
