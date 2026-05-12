import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

class TimeFeature:
    """Base class for time feature extraction."""
    def __init__(self):
        pass
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

class MonthOfYear(TimeFeature):
    """Month of year (1-12) normalized to [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.month.values / 11.0 - 0.5

class DayOfMonth(TimeFeature):
    """Day of month (1-31) normalized to [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.day.values / 30.0 - 0.5

class DayOfWeek(TimeFeature):
    """Day of week (0-6) normalized to [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek.values / 6.0 - 0.5

class HourOfDay(TimeFeature):
    """Hour of day (0-23) normalized to [-0.5, 0.5]."""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour.values / 23.0 - 0.5

def time_features(index: pd.DatetimeIndex, freq: str = 'h') -> np.ndarray:
    """
    Extract time features from datetime index.
    
    Args:
        index: DatetimeIndex to extract features from
        freq: Frequency of the data ('h','t','s','m','w','d','b')
    
    Returns:
        Array of shape [num_features, len(index)] with normalized features
    """
    features_map = {
        'h': [HourOfDay, DayOfWeek, DayOfMonth, MonthOfYear],
        't': [HourOfDay, DayOfWeek, DayOfMonth, MonthOfYear],
        's': [HourOfDay, DayOfWeek, DayOfMonth, MonthOfYear],
        'm': [MonthOfYear],
        'w': [DayOfWeek, MonthOfYear],
        'd': [DayOfWeek, DayOfMonth, MonthOfYear],
        'b': [DayOfWeek, DayOfMonth, MonthOfYear],
    }
    freq = freq.lower()
    if freq not in features_map:
        raise ValueError(f"Unsupported frequency: {freq}")
    
    return np.vstack([feat()(index) for feat in features_map[freq]])
