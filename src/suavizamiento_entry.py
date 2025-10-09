

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns
from scipy.spatial import cKDTree 

import json 
import sys 

from rmprocs.dm import *
from rmprocs.suavizamiento import *


oDmApp, PROJECT_FOLDER = get_oDmApp()

def log(x): 
    x = x + "\n"
    oDmApp.ControlBars.Output.Write(x)

#####################################################

version = 7

args = json.loads(sys.argv[1])
bm = args["bm"]
col = args["col"]
out_bm = args["out_bm"]
out_col = args["out_col"]


response = dict()

#####################################################

log(f"Suavizamiento V{version}\n")

# read the block model
command = f"""
SELCOP 
&IN={bm} &OUT=xxxBMF 
*F1=XC *F2=YC *F3=ZC 
*F4=XINC *F5=YINC *F6=ZINC 
*F7={col}
""".replace("\n", " ")

oDmApp.parseCommand(command)

df = get_dm_table("xxxBMF", oDmApp)

command = "DELETE &IN=xxxBMF"
oDmApp.parseCommand(command)

df[["XC", "YC", "ZC"]] = df[["XC", "YC", "ZC"]].astype("float32")
df["VOL"] = df["XINC"] * df["YINC"] * df["ZINC"] 


if version == 1: 

    dist = int(args["dist"])
    out_cols = [out_col]

    suavizar_col(df, col, dist, out_col)

elif version == 2: 

    dists = [int(d) for d in args["dist"].split(",")]
    out_cols = [f"{out_col}_{d}" for d in dists]

    suavizar_multiple(df, col, dists, out_cols)

elif version == 3: 

    dist = int(args["dist"])
    out_cols = [out_col]

    suavizar_col_batched(
        df, col, dist, out_col, 
        log=log
    )

elif version == 4: 

    dists = [int(d) for d in args["dist"].split(",")]
    out_cols = [f"{out_col}_{d}" for d in dists]

    suavizar_batched_multi(
        df, col, dists, out_cols, 
        log=log
    )

elif version == 5: 

    dist = int(args["dist"])
    out_cols = [out_col]

    suavizar_col_batched_xyz(
        df, col, dist, out_col, 
        x_size=100, y_size=100, z_size=50,
        log=log
    )

elif version == 6: 

    dists = [int(d) for d in args["dist"].split(",")]
    out_cols = [f"{out_col}_{d}" for d in dists]

    suavizar_batched_xyz_multi(
        df, col, dists, out_cols, 
        x_size=50, y_size=50, z_size=50,
        log=log
    )

elif version == 7: 

    dists = [int(d) for d in args["dist"].split(",")]
    out_cols = [f"{out_col}_{d}" for d in dists]

    max_dist = 50 # 1*max(dists)

    suavizar_batched_xyz_multi_stable(
        df, col, dists, out_cols, 
        x_size=max_dist, y_size=max_dist, z_size=max_dist,
        log=log
    )


# reports: mapping of report names to its json table
response["reports"] = dict()


# get the report of volume changes
tvr, tvrf = reportar_volumenes(df, col, dists, out_cols)
response["reports"]["TOTAL VOLUMES"] = df_to_json_table(tvrf)
save_df_as_table(tvr, f"{bm}_total_volumnes", oDmApp)


# for each distance, get the orig vs suav_dist report table
for out_col, dist in zip(out_cols, dists):
    report_name = f"VOLUME ORIG vs {dist}"
    report =  report_volume_variation(df, "VOL", col, out_col)
    response["reports"][report_name] = df_to_json_table(report)
    save_df_as_table(report, f"{bm}_volume_orig_vs_{dist}", oDmApp)


# construct a list with all the available reports
response["reports_list"] = list(response["reports"].keys())

msg = f"Writing output"
oDmApp.ControlBars.Output.write(msg)

command = f"EXTRA &IN={bm} &OUT=xxx1 'GO'"
oDmApp.parseCommand(command)

save_df_as_table(df[out_cols], "xxx2", oDmApp)

command = f"SPLAT &IN1=xxx1 &IN2=xxx2 &out={out_bm}"
oDmApp.parseCommand(command)

command = "DELETE &IN=xxx1"
oDmApp.parseCommand(command)

command = "DELETE &IN=xxx2"
oDmApp.parseCommand(command)

#####################################################

response["status"] = "OK"
response = json.dumps(response)

print(response)
