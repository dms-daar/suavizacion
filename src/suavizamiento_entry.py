

import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns
from scipy.spatial import cKDTree 

import json 
import sys 

from rmprocs.dm import *
from rmprocs.suavizamiento import *

args = json.loads(sys.argv[1])

version = 4

bm = args["bm"]
col = args["col"]

out_bm = args["out_bm"]
out_col = args["out_col"]

#####################################################

oDmApp, PROJECT_FOLDER = get_oDmApp()

def log(x): 
    x = x + "\n"
    oDmApp.ControlBars.Output.Write(x)

log(f"Suavizamiento V{version}\n")

# read the block model
df = get_dm_table(bm, oDmApp)
df = df[["XC", "YC", "ZC", col]]
df[["XC", "YC", "ZC"]] = df[["XC", "YC", "ZC"]].astype("float32")


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
    suavizar_col_batched(df, col, dist, out_col, log=log)

elif version == 4: 
    dists = [int(d) for d in args["dist"].split(",")]
    out_cols = [f"{out_col}_{d}" for d in dists]
    suavizar_batched_multi(df, col, dists, out_cols, log=log)


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

response = json.dumps({
    "status": "OK"
})

print(response)
