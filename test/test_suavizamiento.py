

from fixtures import * 

from rmprocs.dm import *
from rmprocs.suavizamiento import *


# def test_suavizar_col(oDmApp): 

#     table = "mb_final_smooth_r3a"
#     col = "CATE"
#     dist = 20

#     out_col = f"{col}_SUAV"

#     df1 = get_dm_table(table, oDmApp)
#     suavizar_col(df1, col, dist, out_col)

#     assert out_col in df1.columns
#     assert (df1[col] != df1[out_col]).any()


# def test_suavizar_multiple(oDmApp): 

#     table = "mb_final_smooth_r3a"
#     col = "CATE"

#     df10 = get_dm_table(table, oDmApp)
#     df20 = df10.copy()
#     dfm = df10.copy()

#     out_col = f"{col}_SUAV"
#     suavizar_col(df10, col, 10, out_col)
#     suavizar_col(df20, col, 20, out_col)

#     dists = [10, 20]
#     out_cols = [f"{out_col}_{d}" for d in dists]

#     suavizar_multiple(dfm, col, [10, 20], out_cols)

#     assert (df10[out_col] == dfm[f"{out_col}_10"]).all()
#     assert (df20[out_col] == dfm[f"{out_col}_20"]).all()


# def test_suavizamiento_batched(oDmApp):

#     table = "mb_final_smooth_r3a"
#     col = "CATE"
#     dist = 20
#     out_col = f"{col}_SUAV"

#     df = get_dm_table(table, oDmApp)

#     df1 = df.copy()
#     df2 = df.copy()

#     suavizar_col(df1, col, dist, out_col)
#     suavizar_col_batched(df2, col, dist, out_col)

#     assert (df1[out_col] == df2[out_col]).all()


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
