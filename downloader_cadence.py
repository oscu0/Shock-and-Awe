"""Native-epoch integrity and minimum cadence for cleaned satellite products."""

import numpy as np
import pandas as pd


def unique_epoch_rows(frame, source):
    """Collapse identical repeated observations; reject conflicting same-time rows.

    Adjacent inclusive CDAS requests can return the same endpoint twice. Keep
    one observation only when every column, including missingness, agrees.
    Never average distinct native observations or select between disagreements.
    """
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{source}: expected a DatetimeIndex before merging.")
    if frame.index.hasnans:
        raise ValueError(f"{source}: missing observation timestamps before merging.")
    repeated = frame.index.duplicated(keep=False)
    if not repeated.any():
        return frame
    values = frame.loc[repeated].groupby(level=0, sort=False).nunique(dropna=False)
    conflicts = values.gt(1).any(axis=1)
    if conflicts.any():
        epochs = values.index[conflicts].astype(str).tolist()
        raise ValueError(f"{source}: conflicting values at duplicate epochs: {epochs[:5]}")
    unique = frame.loc[~frame.index.duplicated(keep="first")]
    print(f"{source}: removed {len(frame)-len(unique)} identical repeated epoch rows.")
    return unique


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


def merge_products(products, *, average=(), index_name="Epoch", drop_empty=True):
    """Outer-join already cleaned products on their independent UTC epochs.

    ``products`` maps diagnostic source labels to frames in the intended column
    order. Only labels explicitly listed in ``average`` pass through the shared
    sub-second averaging policy. Leave coordinates and quality flags at native
    cadence. No interpolation or nearest-time filling is performed.

    Source-specific cleanup, units, rotations and metadata belong at the call
    site. Inputs are copied, identical repeated epochs are collapsed before
    averaging, and conflicting same-time observations raise an error.
    """
    unknown = set(average) - products.keys()
    if unknown:
        raise ValueError(f"Unknown products selected for averaging: {sorted(unknown)}")

    prepared = []
    for source, frame in products.items():
        product = frame.copy()
        product.index = pd.to_datetime(product.index, utc=True)
        product.index.name = index_name
        product = unique_epoch_rows(product, source)
        if source in average:
            product = average_subsecond(product)
        prepared.append(product)

    merged = pd.concat(prepared, axis=1).sort_index()
    return merged.dropna(how="all") if drop_empty else merged
