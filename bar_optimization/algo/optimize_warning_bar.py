import logging
from itertools import combinations
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from utility.algo.optimize_capacity import OptimizeCapacity
from warning.algo.loadwarning import get_warning


class OptimizeWarningBar:
    def __init__(
        self,
        df: pd.DataFrame,
        battery_capacity: float,
        battery_initial: float,
        contract_capacity: float,
        threshold: List[float] = [0.9, 0.85, 0.8, 0.3],
        discharging_range: List[int] = [7, 5, 3],
        charging_range: List[int] = [5],
        charging_period: List[int] = [0, 5],
        **kwargs: Any,
    ) -> None:

        self.df = df
        self.battery_capacity = battery_capacity
        self.battery_initial = battery_initial
        self.contract_capacity = contract_capacity
        self.threshold = threshold
        self.discharging_range = discharging_range
        self.charging_range = charging_range
        self.charging_period = charging_period
        self.safty_point = kwargs.get('safty_point', 0.3)

    def level_to_action(self, level: str) -> Tuple:

        discharging_list = sorted(self.discharging_range, reverse=True)
        charging_list = sorted(self.charging_range, reverse=True)

        if level == 'red':
            return 'discharging', -discharging_list[0]
        elif level == 'yellow':
            return 'discharging', -discharging_list[1]
        elif level == 'green':
            return 'discharging', -discharging_list[2]
        elif level == 'white':
            return 'charging', charging_list[0]

        return 'standby', 0

    def simulate_data(self) -> pd.DataFrame:

        '''
        df must include time, load and solar columns.
        '''

        check_cols = ['time', 'load', 'solar']
        for i in check_cols:
            if i not in self.df.columns:
                raise ValueError("'time', 'load', 'solar' must be included")

        data = self.df.copy()
        battery_kwh = self.battery_initial
        battery_capacity = self.battery_capacity

        # charging_period:
        start_charging_index = (
            data['time'].dt.time >= pd.to_datetime(f'{str(self.charging_period[0])}:00').time()
        )
        end_charging_index = (
            data['time'].dt.time <= pd.to_datetime(f'{str(self.charging_period[1])}:00').time()
        )

        label = get_warning(
            data['load'] - data['solar'], threshold=self.threshold, contract_capacity=self.contract_capacity
        )
        label.loc[start_charging_index & end_charging_index, 'level'] = 'white'

        data.loc[:, 'label'] = label['level']

        level_action = data['label'].map(self.level_to_action)
        data['battery'] = np.nan

        for row in pd.DataFrame(level_action).itertuples():
            data.loc[row.Index, 'action'] = row.label[0]
            data.loc[row.Index, 'action_kw'] = row.label[1]
            data.loc[row.Index, 'action_kwh'] = row.label[1] / 4

            # cond1 => 電池容量不能超過額度
            # cond2 => am 0 ~ am 5 充電
            new_battery = battery_kwh + row.label[1] / 4
            if (new_battery <= battery_capacity) & (new_battery >= 0):
                battery_kwh += row.label[1] / 4
            else:
                if new_battery >= 0:
                    battery_kwh += min(battery_capacity - battery_kwh, row.label[1] / 4)
                else:
                    battery_kwh -= battery_kwh

            data.loc[row.Index, 'battery'] = battery_kwh

        data.loc[:, 'delta_battery'] = data.battery.diff()
        data.loc[0, 'delta_battery'] = data.loc[0, 'action_kwh']

        return data

    def optimize_warning_bar(self, **kwargs: Any) -> float:

        summer_price = kwargs.get('summer_price', 236.2)
        non_summer_price = kwargs.get('non_summer_price', 173.2)

        # 資料集涵蓋時間、負載、太陽能，通過不同電池行為，決定 load_kw 為多少，接著計算一般電費
        tmp = self.simulate_data()
        tmp.loc[:, 'load_kw'] = tmp['load'] - tmp['solar'] + tmp['delta_battery'] * 4
        max_kw = tmp.set_index('time').resample('M')['load_kw'].max()
        max_kw = max_kw[max_kw > self.contract_capacity]
        if len(max_kw) >= 1:  # 如果超約，計算超約的月份即可
            month = [i if i != 13 else 1 for i in max_kw.index.month + 1]
            oc = OptimizeCapacity()
            before_result = oc.optimize_capacity(
                demand=max_kw,
                capacity=self.contract_capacity,
                summer_price=summer_price,
                non_summer_price=non_summer_price,
                month=month,
            )
            return np.round(before_result['original_fee'][0], 3)
        return 0.0

    def obj_fn(self, bar: List[float]) -> float:
        comb = pd.Series(bar)
        logging.info('combination:', np.array(comb))
        bar_model = OptimizeWarningBar(
            df=self.df,
            battery_capacity=self.battery_capacity,
            battery_initial=self.battery_initial,
            contract_capacity=self.contract_capacity,
            threshold=comb,
            discharging_range=self.discharging_range,
            charging_range=self.charging_range,
            charging_period=self.charging_period,
        )
        fitness = bar_model.optimize_warning_bar()
        logging.info('fitness:', fitness)
        return fitness

    def get_optimal_result(self, **kwargs: Any) -> pd.DataFrame:

        epoch = kwargs.get('epoch', 50)
        threshold_all = pd.Series(list(combinations(np.arange(0.81, 0.99, 0.01).round(2), 3)))
        rnd_idx = np.random.choice(np.arange(len(threshold_all)), epoch)
        threshold_tmp = pd.Series(
            [(list(i) + [self.safty_point]) for i in threshold_all[rnd_idx]], dtype=float
        )
        old_result = self.obj_fn(self.threshold)
        result = threshold_tmp.map(self.obj_fn)
        result = pd.concat((result, threshold_tmp), axis=1).rename(columns={0: 'new_fees', 1: "bar"})
        result = result.sort_values(['new_fees', 'bar'], ascending=[True, False])
        result['old_fees'] = old_result
        result['save'] = result['old_fees'] - result['new_fees']
        result['judge'] = [False if i == 0 else True for i in result['save']]

        return result
