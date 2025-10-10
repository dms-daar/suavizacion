

import pandas as pd
import numpy as np
import win32com.client
import os 
import pywintypes
from datetime import datetime


def getDmFileType():
    oDmTable = win32com.client.Dispatch("DmFile.DmTable")
    strDMext = ".dm"
    try:
        oDmTable.DefaultDatamineFormat
        strDMext = ".dmx"
    except Exception as e:
        pass 
    return strDMext
 

DMEXT = getDmFileType()


def get_oDmApp(app="StudioRM"): 

    oDmApp = None
    
    try:
        oDmApp = win32com.client.Dispatch(f'Datamine.{app}.Application')
    except pywintypes.com_error as e:
        raise Exception(f'No {app} application detected')

    assert oDmApp.ActiveProject is not None, 'No ActiveProject detected'

    # create the temp folder if not exists 
    if not os.path.isdir(rf"C:\temp"): 
        os.mkdir(rf"C:\temp")

    PROJECT_FOLDER = oDmApp.ActiveProject.Folder 
    
    return oDmApp, PROJECT_FOLDER


def get_dm_table(table, oDmApp, columns=None):

    project_folder = oDmApp.ActiveProject.Folder
    table_path = os.path.join(project_folder, f"{table}{DMEXT}")
    assert os.path.exists(table_path), f"Table {table}{DMEXT} does not exist"

    # --- timestamped temp path ---
    temp_dir = r"C:\temp"
    os.makedirs(temp_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(temp_dir, f"temp0_{ts}.csv")

    try:

        if columns is not None: 
            col_sel = [f"*F{i}={col}" for i, col in enumerate(columns, 1)]
            col_sel = " ".join(col_sel)
            command = f"SELCOP &IN={table} &OUT=xxx_{ts} {col_sel}"
            oDmApp.parseCommand(command)
            # Export to CSV
            command = f"output &IN=xxx_{ts} @CSV=1 @NODD=0 @DPLACE=-1 @IMPLICIT=1 '{path}'"
            oDmApp.ParseCommand(command)
            oDmApp.ParseCommand(f"DELETE &IN=xxx_{ts}")


        else: 
            # Export to CSV
            command = f"output &IN={table} @CSV=1 @NODD=0 @DPLACE=-1 @IMPLICIT=1 '{path}'"
            oDmApp.ParseCommand(command)

        # Read and return
        df = pd.read_csv(path, encoding="1252")
        return df

    finally:
        # Always attempt cleanup
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            # optional: log cleanup issue
            pass


def save_df_as_table(df, table, oDmApp):
    # --- timestamped temp path ---
    temp_dir = r"C:\temp"
    os.makedirs(temp_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(temp_dir, f"temp1_{ts}.csv")

    # Work on a copy
    temp_df = df.copy()

    try:
        # normalize column names (no spaces)
        temp_df.columns = [c.replace(" ", "_") for c in temp_df.columns]

        columns = temp_df.columns
        numeric_columns = temp_df.select_dtypes(include='number').columns
        object_columns = temp_df.columns.difference(numeric_columns)

        # cast non-numeric columns to str and compute max lengths
        temp_df[object_columns] = temp_df[object_columns].astype(str)
        object_columns_length = {
            c: int(temp_df[c].str.len().max()) if len(temp_df) else 0
            for c in object_columns
        }

        # write CSV without header/index
        temp_df.to_csv(path, index=False, header=False)

        # build INPFIL column description
        columns_desc = "\n".join([
            (f"'{c} N Y -'" if c in numeric_columns
             else f"'{c} A {object_columns_length.get(c, 0)} Y -'")
            for c in columns
        ])

        command = f"""
        INPFIL &OUT={table} @PRINT=0
        'description'
        {columns_desc}
        '!'
        'OK'
        '{path}'
        """

        # send command (single-line)
        oDmApp.ParseCommand(command.replace("\n", ""))

    finally:
        # always try to delete the temp file
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            # optional: log/ignore cleanup failure
            pass


def load_3d(table, oDmApp, template=None):

    objVR = oDmApp.ActiveProject.VR

    # cargar el archivo file a la ventana 3D
    object = oDmApp.ActiveProject.Data.LoadFromProject(table)    

    if template: 
        
        try: 
            # aplicar el template al objeto
            objOverlay = objVR.Overlays.GetAtName(object.name)
            objVR.ApplyTemplate(objOverlay, template)
            
        except Exception as e: 
            pass 

    # retornar el objeto cargado a la ventana 3D
    return object


def add_cols(table, df, out, oDmApp): 

    # make a copy of the table
    command = f"EXTRA &IN={table} &OUT=xxx1 'GO'"
    oDmApp.parseCommand(command)

    # save the df
    save_df_as_table(df, "xxx2", oDmApp)

    # splat
    command = f"SPLAT &IN1=xxx1 &IN2=xxx2 &out={out}"
    oDmApp.parseCommand(command)

    # delete temp files
    command = "DELETE &IN=xxx1"
    oDmApp.parseCommand(command)

    command = "DELETE &IN=xxx2"
    oDmApp.parseCommand(command)
