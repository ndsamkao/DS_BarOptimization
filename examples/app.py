from bar_optimization.algo.optimize_warning_bar import OptimizeWarningBar
import pandas as pd
  
data = pd.read_csv('bar_optimization/data/optimize_warning_bar_data.csv', parse_dates=['time'])

bar_model = OptimizeWarningBar(df = data, 
                 battery_capacity = 15, 
                 battery_initial = 0, 
                 contract_capacity = 120, 
                 threshold = [0.9, 0.8, 0.7, 0.3],
                 discharging_range = [7, 5, 3],
                 charging_range = [5],
                 charging_period = [0, 5])

result = bar_model.get_optimal_result(epoch=3)
print(result)