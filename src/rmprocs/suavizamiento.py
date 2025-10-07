

from scipy.spatial import cKDTree 
import pandas as pd 
import numpy as np


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

    return df 


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

    return df


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
            leafsize=64,
            balanced_tree=False,
            compact_nodes=False,
            copy_data=False
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
    return df
