

from fixtures import * 
from test_functions import *

from rmprocs.dm import *
from rmprocs.suavizamiento import *

from pandas.testing import assert_series_equal 


def test_suavizar_col(oDmApp): 

    table = "mb_final_smooth_r3a"
    col = "CATE"
    dist = 20

    out_col = f"{col}_SUAV"

    df1 = get_dm_table(table, oDmApp)
    suavizar_col(df1, col, dist, out_col)

    assert out_col in df1.columns
    assert (df1[col] != df1[out_col]).any()


def test_suavizar_multiple(oDmApp): 

    table = "mb_final_smooth_r3a"
    col = "CATE"

    df10 = get_dm_table(table, oDmApp)
    df20 = df10.copy()
    dfm = df10.copy()

    out_col = f"{col}_SUAV"
    suavizar_col(df10, col, 10, out_col)
    suavizar_col(df20, col, 20, out_col)

    dists = [10, 20]
    out_cols = [f"{out_col}_{d}" for d in dists]

    suavizar_multiple(dfm, col, [10, 20], out_cols)

    assert (df10[out_col] == dfm[f"{out_col}_10"]).all()
    assert (df20[out_col] == dfm[f"{out_col}_20"]).all()


def test_suavizamiento_batched1(oDmApp):

    table = "mb_final_smooth_r3a"
    col = "CATE"
    dist = 20
    out_col = f"{col}_SUAV"

    df = get_dm_table(table, oDmApp)

    df1 = df.copy()
    df2 = df.copy()

    suavizar_col(df1, col, dist, out_col)
    suavizar_col_batched(df2, col, dist, out_col, level_thickness=10)

    assert (df1[out_col] == df2[out_col]).all()


def test_suavizamiento_batched1(oDmApp):

    table = "mb_final_smooth_r3a"
    col = "CATE"
    dist = 20
    out_col = f"{col}_SUAV"

    df = get_dm_table(table, oDmApp)

    df1 = df.copy()
    df2 = df.copy()

    suavizar_col(df1, col, dist, out_col)
    suavizar_col_batched(df2, col, dist, out_col, level_thickness=50)

    assert (df1[out_col] == df2[out_col]).all()


def test_suavizar_batched_multi(oDmApp): 

    table = "mb_final_smooth_r3a"
    col = "CATE"
    out_col = f"{col}_SUAV"

    df10 = get_dm_table(table, oDmApp)
    df20 = df10.copy()
    dfm = df10.copy()

    suavizar_col(df10, col, 10, out_col)
    suavizar_col(df20, col, 20, out_col)

    dists = [10, 20]
    out_cols = [f"{out_col}_{d}" for d in dists]
    
    suavizar_batched_multi(
        dfm, col, 
        dists, out_cols
    )

    assert (dfm[f"{out_col}_10"] == df10[out_col]).all()
    assert (dfm[f"{out_col}_20"] == df20[out_col]).all()


def test_suavizar_col_batched_xyz_1(oDmApp): 

    table = "mb_final_smooth_r3a"
    col = "CATE"
    dist = 20
    out_col = f"{col}_SUAV"

    df1 = get_dm_table(table, oDmApp)
    df2 = df1.copy()

    suavizar_col(df1, col, dist, out_col)

    suavizar_col_batched_xyz(
        df2, col, dist, out_col, 
        x_size=10, y_size=10, z_size=10
    )

    assert (df1[out_col] == df2[out_col]).all()


def test_suavizar_batched_xyz_multi_1(oDmApp): 

    table = "mb_final_smooth_r3a"
    col = "CATE"
    out_col = f"{col}_SUAV"

    df10 = get_dm_table(table, oDmApp)
    df20 = df10.copy()
    dfm = df10.copy()

    suavizar_col(df10, col, 10, out_col)
    suavizar_col(df20, col, 20, out_col)

    dists = [10, 20]
    out_cols = [f"{out_col}_{d}" for d in dists]

    suavizar_batched_xyz_multi(
        dfm, col, dists, out_cols, 
        x_size=50, y_size=50, z_size=50
    )

    assert (dfm[f"{out_col}_10"] == df10[out_col]).all()
    assert (dfm[f"{out_col}_20"] == df20[out_col]).all()


def test_suavizar_batched_xyz_multi_stable_1(oDmApp): 

    table = "mb_final_smooth_r3a"
    col = "CATE"
    out_col = f"{col}_SUAV"

    df10 = get_dm_table(table, oDmApp)
    df20 = df10.copy()
    dfm = df10.copy()

    suavizar_col(df10, col, 10, out_col)
    suavizar_col(df20, col, 20, out_col)

    dists = [10, 20]
    out_cols = [f"{out_col}_{d}" for d in dists]

    suavizar_batched_xyz_multi_stable(
        dfm, col, dists, out_cols, 
        x_size=50, y_size=50, z_size=50
    )

    # dfm[f"{out_col}_10"] = dfm[f"{out_col}_10"].astype(int)
    # dfm[f"{out_col}_20"] = dfm[f"{out_col}_20"].astype(int)

    # assert (dfm[f"{out_col}_10"] == df10[out_col]).all()
    # assert (dfm[f"{out_col}_20"] == df20[out_col]).all()

    res, _ = series_equal_with_tolerance(
        dfm[f"{out_col}_10"], 
        df10[out_col], 
        max_diff_pct=0.1 # 1=1% 0.1=0.1%
    )

    assert res, f"Error"

