import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def series_equal_with_tolerance(
    s1: pd.Series,
    s2: pd.Series,
    max_diff_pct: float = 0.1,   # percent of allowed mismatches (e.g., 0.1 = 0.1%)
    align_on: str = "index",     # "index" (inner join) or "position"
    ignore_na: bool = True,      # ignore pairs where either side is NA
    rtol: float = 0.0,           # numeric relative tolerance (for floats/ints)
    atol: float = 0.0            # numeric absolute tolerance
):
    """
    Returns (is_equal: bool, stats: dict)
    stats = {"n_compared", "n_diff", "diff_pct", "n_total_pairs", "n_skipped"}
    """
    if not isinstance(s1, pd.Series) or not isinstance(s2, pd.Series):
        raise TypeError("Both inputs must be pandas Series")

    # Align
    if align_on == "index":
        a, b = s1.align(s2, join="inner")
    elif align_on == "position":
        n = min(len(s1), len(s2))
        a, b = s1.iloc[:n], s2.iloc[:n]
    else:
        raise ValueError("align_on must be 'index' or 'position'")

    # Handle NA selection
    if ignore_na:
        valid = a.notna() & b.notna()
    else:
        # Treat NaN==NaN as equal
        both_na = a.isna() & b.isna()
        valid = ~(a.isna() ^ b.isna())  # only drop pairs where exactly one is NA

    # Nothing to compare?
    n_total_pairs = len(a)
    n_skipped = (n_total_pairs - valid.sum())
    if valid.sum() == 0:
        stats = dict(n_compared=0, n_diff=0, diff_pct=0.0,
                     n_total_pairs=n_total_pairs, n_skipped=int(n_skipped))
        # With no comparable entries, consider them equal by convention
        return True, stats

    av = a[valid]
    bv = b[valid]

    # Comparison rule
    if is_numeric_dtype(av) and is_numeric_dtype(bv):
        equal_mask = np.isclose(av.to_numpy(), bv.to_numpy(), rtol=rtol, atol=atol, equal_nan=True)
    else:
        # Fallback to exact equality for non-numerics (strings, categoricals, etc.)
        equal_mask = (av.values == bv.values)

    n_compared = int(valid.sum())
    n_diff = int((~equal_mask).sum())
    diff_pct = (n_diff / n_compared) * 100.0

    is_equal = diff_pct <= max_diff_pct
    stats = dict(n_compared=n_compared, n_diff=n_diff, diff_pct=diff_pct,
                 n_total_pairs=n_total_pairs, n_skipped=int(n_skipped))
    return is_equal, stats

