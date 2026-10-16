"""Minimum cadence for cleaned satellite magnetic fields and plasma moments."""

import numpy as np
import pandas as pd


def average_subsecond(frame):
    """Average a numeric source product only when its median cadence is below 1 s.

    Call after fill/quality cleanup and before combining products with different
    time axes. UTC bins are labelled at their left edge, [t, t + 1 s). Only bins
    containing source timestamps are emitted; NaNs are ignored in means and
    empty time intervals are never filled. Products at 1 s or slower are returned
    unchanged. Error columns retain the mean source uncertainty, not a SEM.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Science products must have a DatetimeIndex before averaging.")
    if frame.index.hasnans:
        raise ValueError("Science timestamps must be valid before averaging.")
    times = frame.index.unique().sort_values().as_unit("ns").asi8
    intervals = np.diff(times)
    if not len(intervals) or np.median(intervals) >= 1_000_000_000:
        return frame
    if not all(pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
               for dtype in frame.dtypes):
        raise TypeError("Average only numeric science columns after removing quality flags.")
    averaged = frame.groupby(frame.index.floor("s"), sort=True).mean()
    averaged.index.name = frame.index.name
    averaged.attrs = frame.attrs.copy()
    return averaged
