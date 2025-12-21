import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

class TimeFeature:
    def __init__(self):
        pass
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.month.values / 11.0 - 0.5

class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.day.values / 30.0 - 0.5

class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek.values / 6.0 - 0.5

class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour.values / 23.0 - 0.5

def time_features(index, freq='h'):
    features_map = {
        'h': [HourOfDay, DayOfWeek, DayOfMonth, MonthOfYear],
        't': [HourOfDay, DayOfWeek, DayOfMonth, MonthOfYear],
        's': [HourOfDay, DayOfWeek, DayOfMonth, MonthOfYear],
        'm': [MonthOfYear],
        'w': [DayOfWeek, MonthOfYear],
        'd': [DayOfWeek, DayOfMonth, MonthOfYear],
        'b': [DayOfWeek, DayOfMonth, MonthOfYear],
    }
    return np.vstack([feat()(index) for feat in features_map[freq.lower()]])