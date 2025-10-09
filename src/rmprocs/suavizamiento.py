

from scipy.spatial import cKDTree 
import pandas as pd 
import numpy as np
import json, re


def suavizar_col(df, col, dist, out_col): 

    X = df[["XC", "YC", "ZC"]] 
    c = df[col].values

    # create a KDTree with the available available_points
    tree = cKDTree(
        X.values,
        leafsize=64,
        balanced_tree=False,   # slightly less work/overhead
        compact_nodes=False,   # skip compaction pass
        copy_data=False
    )

    indices = tree.query_ball_point(X, r=dist)

    total = len(X)
    for i in range(total):

        # if i % 1000 == 0:
        #     msg = f"Processed [{i}/{total}] \n"
        #     oDmApp.ControlBars.Output.write(msg)

        idx = X.index[i]

        neighbours_indices = indices[i]
        neighbours_anom = c[neighbours_indices]
        imputed_value = pd.Series(neighbours_anom).mode().max()

        df.loc[idx, out_col] = imputed_value


def suavizar_multiple(df, col, dists, out_cols):

    assert len(dists) == len(out_cols)

    available_points = df[["XC", "YC", "ZC"]]
    available_values = df[col].values

    # n_cells = len(available_points)
    # msg = f"indexing {n_cells} cells"
    # oDmApp.ControlBars.Output.write(msg)

    # create a KDTree with the available available_points
    tree = cKDTree(
        available_points.values,
        leafsize=64,
        balanced_tree=False,   # slightly less work/overhead
        compact_nodes=False,   # skip compaction pass
        copy_data=False
    )

    for d, out_col in zip(dists, out_cols):

        indices = tree.query_ball_point(available_points.values, r=d)

        total = len(available_points)
        for i in range(total):

            # if i % 1000 == 0:
            #     msg = f"Dist={d} [{i}/{total}] \n"
            #     oDmApp.ControlBars.Output.write(msg)

            idx = available_points.index[i]

            neighbours_indices = indices[i]
            neighbours_anom = available_values[neighbours_indices]
            imputed_value = pd.Series(neighbours_anom).mode().max()

            df.loc[idx, out_col] = imputed_value


def suavizar_col_batched(
        df, 
        col, 
        dist, 
        out_col, 
        level_thickness=20.0, 
        progress_every=5000, 
        log=lambda x: None
    ):

    # ---- params you choose ----
    # level_thickness = 20.0   # example thickness for each core band
    overlap         = dist   # important: ensure overlap >= dist
    # progress_every  = 1000   # rows

    # convenience arrays
    coords = df[["XC", "YC", "ZC"]].to_numpy(dtype=np.float32, copy=False)
    values = df[col].to_numpy(copy=False)

    zmin = float(df["ZC"].min())
    zmax = float(df["ZC"].max())

    # Define core windows that do not overlap; we only write results for the core.
    # Each KDTree is built on an expanded band: [core_start - overlap, core_end + overlap]
    # Step advances by the core thickness to avoid double-writes.
    level_edges = []
    z_start = zmin
    while z_start <= zmax:
        z_end = z_start + level_thickness
        level_edges.append((z_start, z_end))
        z_start = z_end  # move by full core thickness (no overlap between cores)

    total_cores = len(level_edges)
    log(f"Processing {total_cores} levels (core={level_thickness}, overlap={overlap})")

    for li, (core_lo, core_hi) in enumerate(level_edges, 1):
        # Expanded band for the tree
        exp_lo = core_lo - overlap
        exp_hi = core_hi + overlap

        # Masks / indices
        exp_mask  = (df["ZC"] >= exp_lo) & (df["ZC"] < exp_hi)
        core_mask = (df["ZC"] >= core_lo) & (df["ZC"] < core_hi)

        exp_idx  = df.index[exp_mask]
        core_idx = df.index[core_mask]

        if len(core_idx) == 0:
            continue  # empty core, skip

        # Build KDTree on expanded points (reduces memory vs global tree)
        exp_pts = coords[exp_mask.values]            # (Ne, 3) float32 view
        exp_val = values[exp_mask.values]            # (Ne,)

        # Guard: if expanded band is too small, just write self value
        if len(exp_pts) == 0:
            df.loc[core_idx, out_col] = df.loc[core_idx, col]
            continue

        tree = cKDTree(
            exp_pts,
            # leafsize=64,
            # balanced_tree=False,
            # compact_nodes=False,
            # copy_data=False
        )

        # Query only the core points against the expanded-tree
        core_pts = coords[core_mask.values]  # (Nc, 3)
        neighbors = tree.query_ball_point(core_pts, r=dist)

        # Smooth core points and write results
        # Use mode of neighbor values; fall back to the point's own value if empty.
        # (neighbors[i] are local indices into exp_pts/exp_val)
        Nc = len(core_idx)
        core_out = np.empty(Nc, dtype=values.dtype)

        for i in range(Nc):

            if i % progress_every == 0:
                log(
                    f"Level {li}/{total_cores}  {i}/{Nc} cells (Z:[{core_lo:.2f},{core_hi:.2f}))"
                )

            loc = neighbors[i]
            if not loc:  # no neighbors within radius
                # Use original value of the core point
                core_out[i] = df.loc[core_idx[i], col]
                continue

            # Compute mode; using pandas is simple & robust for non-integers/categories
            vals = pd.Series(exp_val, index=None).iloc[loc]
            # If multiple modes, take the max like your original code
            m = vals.mode()
            core_out[i] = m.max() if len(m) else df.loc[core_idx[i], col]

        # Write only for the core band to avoid double-writes between overlapping bands
        df.loc[core_idx, out_col] = core_out


def suavizar_col_batched_xyz(
    df: pd.DataFrame,
    col: str,
    dist: float,
    out_col: str,
    x_size: float,
    y_size: float,
    z_size: float,
    *,
    leafsize: int = 64,
    progress_every: int = 5000,
    log=lambda msg: None,
):
    """
    KDTree smoothing in 3D tiles (X×Y×Z). For each core tile, we build a KDTree on
    the expanded region (core ± dist in each axis) and write results only for the core.

    Parameters
    ----------
    df : DataFrame
        Must contain ["XC", "YC", "ZC"] and the source column `col`.
    col : str
        Column name to smooth (categorical or numeric; mode is taken).
    dist : float
        Neighborhood radius for smoothing. Also used as overlap in all axes.
    out_col : str
        Output column to write smoothed values into.
    x_size, y_size, z_size : float
        Core tile dimensions along X, Y, Z (must be > 0).
    leafsize : int
        KDTree leaf size.
    progress_every : int
        Log every N cells inside a core tile (0/None disables).
    log : callable
        Logger, e.g., `print` or a custom function.

    Returns
    -------
    DataFrame
        `df` with `out_col` filled for all rows (tiles without neighbors copy original).
    """

    # --- Validation ---
    if not all(s > 0 for s in (x_size, y_size, z_size)):
        raise ValueError("x_size, y_size, and z_size must be > 0")
    for k in ("XC", "YC", "ZC", col):
        if k not in df.columns:
            raise KeyError(f'missing required column "{k}"')

    # Ensure output column exists
    if out_col not in df.columns:
        df[out_col] = np.nan

    # Fast views (avoid copies)
    x = df["XC"].to_numpy(dtype=np.float32, copy=False)
    y = df["YC"].to_numpy(dtype=np.float32, copy=False)
    z = df["ZC"].to_numpy(dtype=np.float32, copy=False)
    vals = df[col].to_numpy(copy=False)
    coords = np.column_stack((x, y, z))  # float32

    # Bounds and overlap
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    ov = float(dist)

    # Build non-overlapping cores in each axis: [lo, hi)
    def edges(lo, hi, step):
        e = []
        cur = lo
        while cur <= hi:
            nxt = cur + step
            e.append((cur, nxt))
            cur = nxt
        return e

    x_edges = edges(xmin, xmax, x_size)
    y_edges = edges(ymin, ymax, y_size)
    z_edges = edges(zmin, zmax, z_size)

    total_tiles = len(x_edges) * len(y_edges) * len(z_edges)
    log(f"3D smoothing in {total_tiles} tiles (core sizes: {x_size}×{y_size}×{z_size}, overlap={ov})")

    tile_idx = 0
    for (x_lo, x_hi) in x_edges:
        for (y_lo, y_hi) in y_edges:
            for (z_lo, z_hi) in z_edges:
                tile_idx += 1

                # Expanded (indexing) box
                ex = (x >= x_lo - ov) & (x < x_hi + ov)
                ey = (y >= y_lo - ov) & (y < y_hi + ov)
                ez = (z >= z_lo - ov) & (z < z_hi + ov)
                exp_mask = ex & ey & ez

                # Core (write zone)
                cxm = (x >= x_lo) & (x < x_hi)
                cym = (y >= y_lo) & (y < y_hi)
                czm = (z >= z_lo) & (z < z_hi)
                core_mask = cxm & cym & czm

                if not core_mask.any():
                    continue  # nothing to write in this tile

                exp_idx = df.index[exp_mask]
                core_idx = df.index[core_mask]
                Nc = core_mask.sum()

                # If expanded is empty, copy originals for the core and continue
                if not exp_mask.any():
                    df.loc[core_idx, out_col] = df.loc[core_idx, col]
                    continue

                exp_pts = coords[exp_mask]
                exp_val = vals[exp_mask]
                core_pts = coords[core_mask]

                # KDTree on expanded box
                tree = cKDTree(
                    exp_pts,
                    # leafsize=leafsize,
                    # balanced_tree=False,
                    # compact_nodes=False,
                    # copy_data=False
                )

                # Query neighbors for all core points
                nbrs = tree.query_ball_point(core_pts, r=float(dist))

                # Compute tile output
                core_out = np.empty(Nc, dtype=vals.dtype)

                # Mode with tie -> max (same behavior as your Z-batched function)
                # Using pandas' mode() for robustness (works with numbers/labels)
                for i in range(Nc):
                    if progress_every and (i % progress_every == 0):
                        log(
                            f"Tile {tile_idx}/{total_tiles} "
                            f"X[{x_lo:.2f},{x_hi:.2f}) Y[{y_lo:.2f},{y_hi:.2f}) Z[{z_lo:.2f},{z_hi:.2f}) "
                            f"{i}/{Nc} cells"
                        )
                    loc = nbrs[i]
                    if not loc:
                        # No neighbors inside radius: keep original
                        core_out[i] = df.loc[core_idx[i], col]
                        continue

                    m = pd.Series(exp_val[loc]).mode()
                    core_out[i] = m.max() if len(m) else df.loc[core_idx[i], col]

                # Write only for the core (avoid double writes across overlaps)
                df.loc[core_idx, out_col] = core_out

    log("3D tile-wise KDTree smoothing complete.")


def suavizar_batched_multi(
    df,
    col,
    dists,                 # e.g. [10, 20, 30]
    out_cols,              # e.g. ["SVOL_10", "SVOL_20", "SVOL_30"]
    level_thickness=20.0,
    progress_every=5000,
    log=lambda x: None,
    leafsize=64,
):
    """
    Batched KDTree smoothing by Z levels, for multiple radii at once.
    For each level, builds a KDTree on [core - overlap, core + overlap],
    where overlap = max(dists). Writes results only for the core band.
    """
    assert len(dists) == len(out_cols), "dists and out_cols must match in length"
    if len(dists) == 0:
        return df

    # Ensure output columns exist
    for oc in out_cols:
        if oc not in df.columns:
            df[oc] = np.nan

    # Prepare arrays (float32 to cut memory)
    coords = df[["XC", "YC", "ZC"]].to_numpy(dtype=np.float32, copy=False)
    values = df[col].to_numpy(copy=False)

    zmin = float(df["ZC"].min())
    zmax = float(df["ZC"].max())
    overlap = float(max(dists))  # important for boundary accuracy

    # Build non-overlapping core windows; each tree uses core ± overlap
    level_edges = []
    z_start = zmin
    while z_start <= zmax:
        z_end = z_start + level_thickness
        level_edges.append((z_start, z_end))
        z_start = z_end

    total_cores = len(level_edges)
    log(f"Processing {total_cores} levels (core={level_thickness}, overlap={overlap})")

    for li, (core_lo, core_hi) in enumerate(level_edges, 1):
        exp_lo = core_lo - overlap
        exp_hi = core_hi + overlap

        exp_mask  = (df["ZC"] >= exp_lo) & (df["ZC"] < exp_hi)
        core_mask = (df["ZC"] >= core_lo) & (df["ZC"] < core_hi)

        exp_idx  = df.index[exp_mask]
        core_idx = df.index[core_mask]

        if len(core_idx) == 0:
            continue

        # KDTree on the expanded band
        exp_pts = coords[exp_mask.values]    # (Ne, 3) float32 view
        exp_val = values[exp_mask.values]    # (Ne,)
        core_pts = coords[core_mask.values]  # (Nc, 3)
        Nc = len(core_idx)

        if len(exp_pts) == 0:
            # Nothing to reference—copy original values for this core
            for oc in out_cols:
                df.loc[core_idx, oc] = df.loc[core_idx, col]
            continue

        tree = cKDTree(
            exp_pts,
            leafsize=leafsize,
            balanced_tree=False,
            compact_nodes=False,
            copy_data=False
        )

        # For each radius/out_col, perform smoothing on the *same* core points
        for r, oc in zip(dists, out_cols):
            neighbors = tree.query_ball_point(core_pts, r=float(r))

            # Prepare output buffer for this column
            core_out = np.empty(Nc, dtype=values.dtype)

            for i in range(Nc):
                if progress_every and (i % progress_every == 0):
                    log(f"Level {li}/{total_cores}  {i}/{Nc} cells  (Z:[{core_lo:.2f},{core_hi:.2f}))  r={r}")

                loc = neighbors[i]
                if not loc:  # no neighbors within r
                    core_out[i] = df.loc[core_idx[i], col]
                    continue

                # Mode of neighbors (ties -> max), same behavior as before
                m = pd.Series(exp_val[loc]).mode()
                core_out[i] = m.max() if len(m) else df.loc[core_idx[i], col]

            # Write only for the core band to avoid double-writes
            df.loc[core_idx, oc] = core_out

    log("Level-wise multi-radius KDTree smoothing complete.")


def suavizar_batched_xyz_multi(
    df: pd.DataFrame,
    col: str,
    dists,                 # e.g. [10, 20, 30]
    out_cols,              # e.g. ["SVOL_10", "SVOL_20", "SVOL_30"]
    x_size: float,
    y_size: float,
    z_size: float,
    *,
    leafsize: int = 64,
    progress_every: int = 5000,
    log=lambda msg: None,
):
    """
    KDTree smoothing in 3D tiles (X×Y×Z) for multiple radii at once.

    For each core tile [x_lo,x_hi)×[y_lo,y_hi)×[z_lo,z_hi), build a KDTree on the
    expanded region (core ± max(dists) along all axes). Then, for each radius r in
    `dists`, perform mode-based smoothing for the *same* set of core points and
    write to the corresponding column in `out_cols`.

    Notes
    -----
    - Overlap used for indexing is ov = max(dists).
    - If a core cell has no neighbors within radius r, its original value is copied.
    - Ties in mode pick the maximum (same as your prior behavior).
    - Intervals are half-open: [lo, hi) in each axis.
    """
    # ----- Validation -----
    if not (len(dists) == len(out_cols) and len(dists) > 0):
        raise ValueError("dists and out_cols must have the same nonzero length")
    if not all(s > 0 for s in (x_size, y_size, z_size)):
        raise ValueError("x_size, y_size, and z_size must be > 0")
    for k in ("XC", "YC", "ZC", col):
        if k not in df.columns:
            raise KeyError(f'missing required column "{k}"')

    # Ensure outputs exist
    for oc in out_cols:
        if oc not in df.columns:
            df[oc] = np.nan

    # Fast views
    x = df["XC"].to_numpy(dtype=np.float32, copy=False)
    y = df["YC"].to_numpy(dtype=np.float32, copy=False)
    z = df["ZC"].to_numpy(dtype=np.float32, copy=False)
    vals = df[col].to_numpy(copy=False)
    coords = np.column_stack((x, y, z))  # float32

    # Bounds and overlap
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    ov = float(max(dists))

    # Build non-overlapping core edges in each axis: [lo, hi)
    def edges(lo, hi, step):
        e = []
        cur = lo
        while cur <= hi:
            nxt = cur + step
            e.append((cur, nxt))
            cur = nxt
        return e

    x_edges = edges(xmin, xmax, x_size)
    y_edges = edges(ymin, ymax, y_size)
    z_edges = edges(zmin, zmax, z_size)

    total_tiles = len(x_edges) * len(y_edges) * len(z_edges)
    log(
        f"3D multi-radius smoothing in {total_tiles} tiles "
        f"(core: {x_size}×{y_size}×{z_size}, overlap={ov}; radii={list(dists)})"
    )

    tile_id = 0
    for (x_lo, x_hi) in x_edges:
        # Precompute per-axis masks to reduce re-evaluation
        ex_x = (x >= x_lo - ov) & (x < x_hi + ov)
        cx_x = (x >= x_lo)     & (x < x_hi)
        for (y_lo, y_hi) in y_edges:
            ex_xy = ex_x & ((y >= y_lo - ov) & (y < y_hi + ov))
            cx_xy = cx_x & ((y >= y_lo)     & (y < y_hi))
            for (z_lo, z_hi) in z_edges:
                tile_id += 1

                exp_mask  = ex_xy & ((z >= z_lo - ov) & (z < z_hi + ov))
                core_mask = cx_xy & ((z >= z_lo)      & (z < z_hi))

                if not core_mask.any():
                    continue  # no cells to write in this tile

                exp_idx  = df.index[exp_mask]
                core_idx = df.index[core_mask]
                Nc = core_mask.sum()

                if not exp_mask.any():
                    # No reference points at all — copy originals for all outputs
                    for oc in out_cols:
                        df.loc[core_idx, oc] = df.loc[core_idx, col]
                    continue

                exp_pts = coords[exp_mask]   # (Ne, 3)
                exp_val = vals[exp_mask]     # (Ne,)
                core_pts = coords[core_mask] # (Nc, 3)

                # Build one KDTree for the expanded region
                tree = cKDTree(
                    exp_pts,
                    # leafsize=leafsize,
                    # balanced_tree=False,
                    # compact_nodes=False,
                    # copy_data=False
                )

                # For each radius/out_col, smooth the *same* core points
                for r, oc in zip(dists, out_cols):
                    nbrs = tree.query_ball_point(core_pts, r=float(r))

                    core_out = np.empty(Nc, dtype=vals.dtype)
                    for i in range(Nc):
                        if progress_every and (i % progress_every == 0):
                            log(
                                f"Tile {tile_id}/{total_tiles} "
                                f"X[{x_lo:.2f},{x_hi:.2f}) Y[{y_lo:.2f},{y_hi:.2f}) Z[{z_lo:.2f},{z_hi:.2f}) "
                                f"{i}/{Nc} cells  r={r}"
                            )
                        loc = nbrs[i]
                        if not loc:
                            core_out[i] = df.loc[core_idx[i], col]
                            continue
                        # Mode of neighbors (ties -> max), pandas handles numeric/label
                        m = pd.Series(exp_val[loc]).mode()
                        core_out[i] = m.max() if len(m) else df.loc[core_idx[i], col]

                    # Write only for the core
                    df.loc[core_idx, oc] = core_out

    log("3D multi-radius KDTree smoothing complete.")
    return df


def suavizar_batched_xyz_multi_stable(
    df: pd.DataFrame,
    col: str,
    dists,                 # e.g. [10, 20, 30]
    out_cols,              # e.g. ["SVOL_10", "SVOL_20", "SVOL_30"]
    x_size: float,
    y_size: float,
    z_size: float,
    *,
    leafsize: int = 64,
    progress_every: int = 10000,
    core_batch: int = 50000,       # process core points in chunks to cap peak RAM
    prefer_bincount: bool = True,  # switch to False if your categories are extremely many
    log=lambda msg: None,
):
    """
    3D tile-wise KDTree smoothing (X×Y×Z) for multiple radii, optimized for memory.

    Key stability tactics:
      - One KDTree per tile built on expanded region (± max(dists)).
      - Process core points in batches (core_batch).
      - For each radius, query neighbors for the current batch only, compute modes, write, discard.
      - Factorize `col` to integer codes once; compute modes on small int arrays.
      - No pandas per-row allocations.

    Ties in mode -> pick the **largest** value (max-on-tie), matching your earlier behavior.
    """
    # ---- Validation ----
    if not (len(dists) == len(out_cols) and len(dists) > 0):
        raise ValueError("dists and out_cols must have the same nonzero length")
    if not all(s > 0 for s in (x_size, y_size, z_size)):
        raise ValueError("x_size, y_size, and z_size must be > 0")
    for k in ("XC", "YC", "ZC", col):
        if k not in df.columns:
            raise KeyError(f'missing required column "{k}"')

    # Ensure outputs exist
    for oc in out_cols:
        if oc not in df.columns:
            df[oc] = np.nan

    # ---- Memory-lean views ----
    x = df["XC"].to_numpy(dtype=np.float32, copy=False)
    y = df["YC"].to_numpy(dtype=np.float32, copy=False)
    z = df["ZC"].to_numpy(dtype=np.float32, copy=False)

    # Factorize values -> integer codes (NaN -> -1)
    codes, uniques = pd.factorize(df[col].to_numpy(), sort=False)
    codes = codes.astype(np.int64, copy=False)   # bincount needs non-negative; we'll filter -1
    n_uniques = int(len(uniques))

    # Bounds and overlap
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))
    ov = float(max(dists))

    # Build non-overlapping core edges [lo, hi)
    def edges(lo, hi, step):
        e = []
        cur = lo
        while cur <= hi:
            e.append((cur, cur + step))
            cur += step
        return e

    x_edges = edges(xmin, xmax, float(x_size))
    y_edges = edges(ymin, ymax, float(y_size))
    z_edges = edges(zmin, zmax, float(z_size))

    total_tiles = len(x_edges) * len(y_edges) * len(z_edges)
    log(f"3D multi-radius smoothing (stable) in {total_tiles} tiles "
        f"(core: {x_size}×{y_size}×{z_size}, overlap={ov}; radii={list(dists)})")

    # Pre-sort radii to be polite to caches; we compute per-radius anyway
    dists = list(dists)
    out_cols = list(out_cols)

    # Helper: compute mode code with max-on-tie from an array of non-negative int codes
    # (caller must have removed -1 codes)
    def mode_code(arr_codes: np.ndarray) -> int:
        if arr_codes.size == 0:
            return -1  # sentinel for "no mode"
        # If categories are huge, bincount with minlength can be big; use unique path:
        if (not prefer_bincount) or (n_uniques > 1_000_000) or (arr_codes.size < 4096):
            u, c = np.unique(arr_codes, return_counts=True)
            # max-on-tie -> take largest u among those with max count
            maxc = c.max()
            return int(u[c == maxc].max())
        # Otherwise bincount (fast, but allocates up to n_uniques)
        bc = np.bincount(arr_codes, minlength=n_uniques)
        maxc = bc.max()
        # pick largest index with that count
        # np.where returns ascending indices; take last
        return int(np.where(bc == maxc)[0][-1])

    tile_id = 0
    for (x_lo, x_hi) in x_edges:
        ex_x = (x >= x_lo - ov) & (x < x_hi + ov)
        cx_x = (x >= x_lo)     & (x < x_hi)
        for (y_lo, y_hi) in y_edges:
            ex_xy = ex_x & ((y >= y_lo - ov) & (y < y_hi + ov))
            cx_xy = cx_x & ((y >= y_lo)     & (y < y_hi))
            for (z_lo, z_hi) in z_edges:
                tile_id += 1

                exp_mask  = ex_xy & ((z >= z_lo - ov) & (z < z_hi + ov))
                core_mask = cx_xy & ((z >= z_lo)      & (z < z_hi))

                if not core_mask.any():
                    continue  # nothing to write

                # Get indices (saves RAM vs materializing masked arrays)
                exp_idx  = np.flatnonzero(exp_mask)
                core_idx = np.flatnonzero(core_mask)
                Nc = core_idx.size
                if exp_idx.size == 0:
                    # no references; copy original for each out column
                    for oc in out_cols:
                        df.loc[core_idx, oc] = df.loc[core_idx, col]
                    continue

                # Build KDTree on expanded points (float32 coords)
                exp_pts = np.empty((exp_idx.size, 3), dtype=np.float32)
                exp_pts[:, 0] = x[exp_idx]
                exp_pts[:, 1] = y[exp_idx]
                exp_pts[:, 2] = z[exp_idx]
                tree = cKDTree(
                    exp_pts,
                    leafsize=leafsize,
                    balanced_tree=False,
                    compact_nodes=False,
                    copy_data=False
                )

                # Core coords (chunked)
                core_pts = np.empty((Nc, 3), dtype=np.float32)
                core_pts[:, 0] = x[core_idx]
                core_pts[:, 1] = y[core_idx]
                core_pts[:, 2] = z[core_idx]

                # Pre-extract codes for expanded region to avoid repeated gathers
                exp_codes = codes[exp_idx]  # int64 (contains -1 for NaN)

                # Process core in batches to cap memory
                for start in range(0, Nc, core_batch):
                    stop = min(start + core_batch, Nc)
                    batch_idx = core_idx[start:stop]
                    batch_pts = core_pts[start:stop]

                    # For each radius, compute and write immediately (no big retention)
                    for r, oc in zip(dists, out_cols):
                        # Neighbor lists for this batch only
                        nbrs = tree.query_ball_point(batch_pts, r=float(r))

                        # Prepare output buffer
                        out_codes = np.empty(stop - start, dtype=np.int64)

                        # Compute modes row-by-row (neighbors per row vary)
                        for i, loc in enumerate(nbrs):
                            if progress_every and ((start + i) % progress_every == 0):
                                log(
                                    f"Tile {tile_id}/{total_tiles} "
                                    f"X[{x_lo:.2f},{x_hi:.2f}) Y[{y_lo:.2f},{y_hi:.2f}) Z[{z_lo:.2f},{z_hi:.2f}) "
                                    f"{start + i}/{Nc} cells  r={r}"
                                )
                            if not loc:  # no neighbors: copy original
                                out_codes[i] = codes[batch_idx[i]]
                                continue

                            neigh_codes = exp_codes[loc]
                            # drop NaN sentinel (-1) if present
                            if (neigh_codes >= 0).any():
                                neigh_codes = neigh_codes[neigh_codes >= 0]
                                mc = mode_code(neigh_codes)
                                if mc >= 0:
                                    out_codes[i] = mc
                                else:
                                    out_codes[i] = codes[batch_idx[i]]
                            else:
                                out_codes[i] = codes[batch_idx[i]]

                        # Map codes back to original dtype and write
                        df.loc[batch_idx, oc] = uniques[out_codes]

                # Drop big temporaries ASAP
                del exp_pts, core_pts, exp_codes, tree

    log("3D multi-radius KDTree smoothing complete (memory-stable).")


def report_volume_variation(bm, ton, col, out_col):

    res1 = bm.groupby([col])[[ton]].sum()
    res1 = res1.reset_index()
    res1[ton] = res1[ton].round(0)
    res1 = res1.rename(columns={ton: "CATE_TOTAL_VOL"})


    res2 = bm.groupby([col, out_col])[[ton]].sum()
    res2 = res2.reset_index()

    res2[out_col] = res2[out_col].astype(int)
    res2[ton] = res2[ton].round(0)
    res2 = res2.rename(columns={ton: "VAR_VOL"})

    res2 = res2.set_index(col).join(
        res1.set_index(col), 
        how="left"
    ).reset_index()

    res2["VAR_PCT"] = (res2["VAR_VOL"] / res2["CATE_TOTAL_VOL"]) * 100
    res2 = res2.drop(columns=["CATE_TOTAL_VOL"])

    # nicely formatted for table rendering
    resf = res2.copy()
    resf["VAR_PCT"] = resf["VAR_PCT"].apply(lambda x: f"{x:.1f}%")

    return res2, resf


def reportar_volumenes(bm, col, dists, out_cols): 

    distancias = [0] + dists
    columns = [col] + out_cols
    df = None

    for idx, (c, d) in enumerate(zip(columns, distancias), 0):

        res = bm.groupby(c).agg({"VOL": "sum"})
        res = res / res.sum() * 100

        res.index.name = col

        if d == 0: 
            res = res.rename(columns={"VOL": "ORIG"})
        else: 
            res = res.rename(columns={"VOL": f"SUAV_{d}"})
        
        if idx == 0: df = res
        else: df = df.join(res, how="outer")

    
    dff = df.copy() 
    dff = dff.applymap(lambda x: f"{x:.2f}%")
    dff = dff.reset_index()

    df = df.reset_index()

    return df, dff


def df_to_json_table(df, percent_cols=None, float_ndigits=1):
    """
    Returns a dict like:
    {
      "columns": [{"key":"cate","label":"CATE"}, ...],
      "rows": [{"cate":1, "cate_suav":1, "var_vol":9850, "pct_var":"66.3%"}, ...]
    }
    - percent_cols: list of column names to format as '##.#%' strings.
    """
    percent_cols = set(percent_cols or [])
    cols = list(df.columns)

    # Make safe keys (no spaces/symbols)
    def make_key(name):
        return re.sub(r'\W+', '_', str(name)).strip('_').lower()

    col_defs = [{"key": make_key(c), "label": str(c)} for c in cols]

    rows = []
    for _, r in df.iterrows():
        row = {}
        for c, cdef in zip(cols, col_defs):
            v = r[c]
            key = cdef["key"]

            if pd.isna(v):
                row[key] = None
                continue

            if c in percent_cols:
                # Accepts either 0..1 or 0..100, outputs e.g. '66.3%'
                try:
                    fv = float(v)
                    if fv <= 1.0:
                        fv *= 100.0
                    row[key] = f"{round(fv, float_ndigits)}%"
                except Exception:
                    row[key] = str(v)
            else:
                if isinstance(v, (np.integer, int)):
                    row[key] = int(v)
                elif isinstance(v, (np.floating, float)):
                    row[key] = float(v)
                else:
                    row[key] = str(v)
        rows.append(row)

    return {"columns": col_defs, "rows": rows}


def summarize_bands_xyz_lean(
    df: pd.DataFrame,
    dist: float,
    x_size: float,
    y_size: float,
    z_size: float,
    keep_empty: bool = False,
) -> pd.DataFrame:
    """
    Memory-lean summary for 3D tiling:
      - Bins points to tiles once
      - Aggregates counts and sums per tile
      - Uses a 3D summed-area table to get n_indexed (core ± dist) in O(1)/tile
      - Computes centroids from tile sums (no per-tile boolean masks)

    Returns one row per tile (drops empty tiles unless keep_empty=True) with:
      tile_id, ix, iy, iz,
      x_lo/x_hi, y_lo/y_hi, z_lo/z_hi,
      x_center/y_center/z_center,
      cx/cy/cz (centroid of core points; NaN if empty),
      n_indexed, n_smoothed
    """
    # ---- Validation ----
    if not all(s > 0 for s in (x_size, y_size, z_size)):
        raise ValueError("x_size, y_size, and z_size must be > 0")
    for k in ("XC", "YC", "ZC"):
        if k not in df.columns:
            raise KeyError(f'missing required column "{k}"')

    # Views (compact dtypes)
    x = df["XC"].to_numpy(dtype=np.float32, copy=False)
    y = df["YC"].to_numpy(dtype=np.float32, copy=False)
    z = df["ZC"].to_numpy(dtype=np.float32, copy=False)

    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    zmin, zmax = float(np.nanmin(z)), float(np.nanmax(z))

    # Build edges to match half-open [lo, hi) tiles used elsewhere
    def build_edges(lo, hi, step):
        edges = []
        cur = lo
        # keep final edge even if mostly empty to mirror previous behavior
        while cur <= hi:
            edges.append(cur)
            cur += step
        edges.append(edges[-1] + step)  # last "hi" edge
        return np.array(edges, dtype=np.float32)

    x_edges = build_edges(xmin, xmax, float(x_size))
    y_edges = build_edges(ymin, ymax, float(y_size))
    z_edges = build_edges(zmin, zmax, float(z_size))

    nx = len(x_edges) - 1
    ny = len(y_edges) - 1
    nz = len(z_edges) - 1

    # Bin each row into (ix, iy, iz). Clip to stay inside.
    ix = np.floor((x - x_edges[0]) / (x_edges[1] - x_edges[0])).astype(np.int32)
    iy = np.floor((y - y_edges[0]) / (y_edges[1] - y_edges[0])).astype(np.int32)
    iz = np.floor((z - z_edges[0]) / (z_edges[1] - z_edges[0])).astype(np.int32)
    np.clip(ix, 0, nx - 1, out=ix)
    np.clip(iy, 0, ny - 1, out=iy)
    np.clip(iz, 0, nz - 1, out=iz)

    # Flatten tile index for efficient aggregation
    flat = (ix * ny + iy) * nz + iz
    n_tiles = nx * ny * nz

    # Aggregate counts and sums per tile (int32 / float64 to keep precision in sums)
    counts = np.zeros(n_tiles, dtype=np.int32)
    sum_x  = np.zeros(n_tiles, dtype=np.float64)
    sum_y  = np.zeros(n_tiles, dtype=np.float64)
    sum_z  = np.zeros(n_tiles, dtype=np.float64)

    np.add.at(counts, flat, 1)
    np.add.at(sum_x, flat, x.astype(np.float64, copy=False))
    np.add.at(sum_y, flat, y.astype(np.float64, copy=False))
    np.add.at(sum_z, flat, z.astype(np.float64, copy=False))

    # Reshape to 3D grids
    counts3 = counts.reshape(nx, ny, nz)
    sumx3   = sum_x.reshape(nx, ny, nz)
    sumy3   = sum_y.reshape(nx, ny, nz)
    sumz3   = sum_z.reshape(nx, ny, nz)

    # Centroids for tiles with points
    with np.errstate(invalid="ignore", divide="ignore"):
        cx3 = sumx3 / counts3
        cy3 = sumy3 / counts3
        cz3 = sumz3 / counts3

    # Compute neighborhood radius in tile units (exact for aligned tiles)
    rx = int(np.ceil(dist / (x_edges[1] - x_edges[0])))
    ry = int(np.ceil(dist / (y_edges[1] - y_edges[0])))
    rz = int(np.ceil(dist / (z_edges[1] - z_edges[0])))

    # 3D summed-area table (integral volume) for O(1) rectangular sums
    # pad with a leading zero plane for inclusive-exclusive sums
    integ = counts3.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
    # Pad to shape (nx+1, ny+1, nz+1)
    pad = np.zeros((nx + 1, ny + 1, nz + 1), dtype=np.int64)
    pad[1:, 1:, 1:] = integ

    # helper to query sum over [x0:x1], [y0:y1], [z0:z1] inclusive (tile indices)
    def rect_sum(x0, x1, y0, y1, z0, z1):
        # convert to +1 indexing for integral volume
        x0p, x1p = x0, x1 + 1
        y0p, y1p = y0, y1 + 1
        z0p, z1p = z0, z1 + 1
        return (
            pad[x1p, y1p, z1p]
            - pad[x0p, y1p, z1p]
            - pad[x1p, y0p, z1p]
            - pad[x1p, y1p, z0p]
            + pad[x0p, y0p, z1p]
            + pad[x0p, y1p, z0p]
            + pad[x1p, y0p, z0p]
            - pad[x0p, y0p, z0p]
        )

    # Build output rows
    rows = []
    tile_id = 0
    for i in range(nx):
        x_lo, x_hi = float(x_edges[i]), float(x_edges[i + 1])
        for j in range(ny):
            y_lo, y_hi = float(y_edges[j]), float(y_edges[j + 1])
            for k in range(nz):
                c = int(counts3[i, j, k])
                if not keep_empty and c == 0:
                    continue

                # neighborhood bounds in tile space for expanded box
                il = max(0, i - rx); ih = min(nx - 1, i + rx)
                jl = max(0, j - ry); jh = min(ny - 1, j + ry)
                kl = max(0, k - rz); kh = min(nz - 1, k + rz)
                n_indexed = int(rect_sum(il, ih, jl, jh, kl, kh))

                # Centroid for core (NaN if empty)
                cx = float(cx3[i, j, k]) if c > 0 else np.nan
                cy = float(cy3[i, j, k]) if c > 0 else np.nan
                cz = float(cz3[i, j, k]) if c > 0 else np.nan

                tile_id += 1
                rows.append({
                    "tile_id": tile_id,
                    "ix": i, "iy": j, "iz": k,
                    "x_lo": x_lo, "x_hi": x_hi,
                    "y_lo": y_lo, "y_hi": y_hi,
                    "z_lo": float(z_edges[k]), "z_hi": float(z_edges[k + 1]),
                    "x_center": (x_lo + x_hi) * 0.5,
                    "y_center": (y_lo + y_hi) * 0.5,
                    "z_center": (z_edges[k] + z_edges[k + 1]) * 0.5,
                    "cx": cx, "cy": cy, "cz": cz,
                    "n_indexed": n_indexed,
                    "n_smoothed": c,
                })

    return pd.DataFrame(rows)


# If you prefer the lean version for stability:
# from your_module import summarize_bands_xyz_lean as summarize_bands_xyz

def _axis_step_estimate(vals: np.ndarray, q: float = 0.1) -> float:
    """Robust estimate of native cell size along one axis."""
    u = np.unique(vals.astype(np.float32, copy=False))
    if u.size < 2:
        return 1.0
    d = np.diff(u)
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(np.quantile(d, q))

def _range_estimate(vals: np.ndarray) -> float:
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    return max(1e-9, vmax - vmin)

def find_max_cubic_tile_size(
    df: pd.DataFrame,
    max_dist: float,
    limit_indexed: int = 100_000,
    *,
    summarize_fn = None,              # default: summarize_bands_xyz (or pass your lean version)
    step_quantile: float = 0.1,       # how we estimate native cell step per axis
    max_iters: int = 24,              # binary search iterations
    keep_empty: bool = False,         # only used if your summarize_fn supports it
    verbose: bool = False,
):
    """
    Returns the largest tile size S to use for x_size=y_size=z_size in
    suavizar_batched_xyz_multi_stable such that max(n_indexed) <= limit_indexed.

    Parameters
    ----------
    df : DataFrame with XC, YC, ZC
    max_dist : float
        The overlap/search radius that defines expanded regions.
    limit_indexed : int
        Hard cap for the maximum 'n_indexed' allowed (default 100k).
    summarize_fn : callable
        Function like summarize_bands_xyz(df, dist, x_size, y_size, z_size)
        that returns a DataFrame with an 'n_indexed' column. If None, we
        assume you have summarize_bands_xyz in scope.
    step_quantile : float
        Quantile used to estimate native cell spacing from unique coordinate diffs.
    max_iters : int
        Max binary search iterations.
    keep_empty : bool
        Passed through if your summarize supports it (ignored otherwise).
    verbose : bool
        Print progress.

    Returns
    -------
    size : float
        The largest cubic tile size S satisfying the constraint.
    stats : dict
        Useful diagnostics: {'max_indexed': int, 'summary_rows': int, 'low': float, 'high': float, 'iters': int}
    """
    if summarize_fn is None:
        # fall back to a name in scope; replace if yours has a different name
        summarize_fn = summarize_bands_xyz

    for k in ("XC", "YC", "ZC"):
        if k not in df.columns:
            raise KeyError(f'missing required column "{k}"')

    x = df["XC"].to_numpy(dtype=np.float32, copy=False)
    y = df["YC"].to_numpy(dtype=np.float32, copy=False)
    z = df["ZC"].to_numpy(dtype=np.float32, copy=False)

    # --- establish search bounds for cubic tile size S ---
    # lower bound: “native” cell size (use the max of axis estimates so S covers at least ~1 cell)
    dx = _axis_step_estimate(x, q=step_quantile)
    dy = _axis_step_estimate(y, q=step_quantile)
    dz = _axis_step_estimate(z, q=step_quantile)
    low = max(dx, dy, dz, 1e-6)

    # upper bound: span of the largest axis (one giant tile)
    rx = _range_estimate(x)
    ry = _range_estimate(y)
    rz = _range_estimate(z)
    high = max(rx, ry, rz)

    # guard: if even a huge tile is under the limit, return high
    def max_indexed_for(size: float) -> tuple[int, int]:
        # Some summarize variants accept keep_empty; try to pass it if available.
        try:
            summary = summarize_fn(df, dist=max_dist, x_size=size, y_size=size, z_size=size, keep_empty=keep_empty)
        except TypeError:
            summary = summarize_fn(df, dist=max_dist, x_size=size, y_size=size, z_size=size)
        mx = int(summary["n_indexed"].max()) if len(summary) else 0
        return mx, len(summary)

    # Binary search assumes monotonic non-decreasing max(n_indexed) as size grows (true in practice).
    # First, check the bounds:
    mx_low, _ = max_indexed_for(low)
    mx_high, _ = max_indexed_for(high)

    if verbose:
        print(f"[init] low={low:.6g} -> max_indexed={mx_low}, high={high:.6g} -> max_indexed={mx_high}")

    # If even the smallest tile exceeds the limit, we can't meet the constraint with cubic tiles.
    if mx_low > limit_indexed:
        return None, {"reason": "min_size_exceeds_limit", "max_indexed": mx_low, "low": low, "high": high, "iters": 0}

    # If largest tile still within limit, take it.
    if mx_high <= limit_indexed:
        return float(high), {"max_indexed": mx_high, "low": low, "high": high, "iters": 0}

    # Binary search
    iters = 0
    best = low
    best_mx = mx_low
    for _ in range(max_iters):
        iters += 1
        mid = (low + high) * 0.5
        mx_mid, _ = max_indexed_for(mid)
        if verbose:
            print(f"[{iters}] size={mid:.6g} -> max_indexed={mx_mid}")

        if mx_mid <= limit_indexed:
            best = mid
            best_mx = mx_mid
            low = mid  # try bigger
        else:
            high = mid # need smaller

        # early exit if interval is tiny
        if (high - low) / max(high, 1e-9) < 1e-3:
            break

    return float(best), {"max_indexed": int(best_mx), "low": low, "high": high, "iters": iters}


