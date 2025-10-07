

import pandas as pd
import numpy as np
import win32com.client
import os 
import pywintypes


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


def get_dm_table(table, oDmApp):

    project_folder = oDmApp.ActiveProject.Folder

    table_path = f"{project_folder}\{table}.dm" 
    assert  os.path.exists(table_path), f"Table {table}.dm does not exist"

    path = rf"C:\temp\temp0.csv"
    if os.path.exists(path): os.remove(path)

    command = f"output &IN={table} @CSV=1 @NODD=0 @DPLACE=-1 @IMPLICIT=1 '{path}'"
    oDmApp.ParseCommand(command)

    df = pd.read_csv(path, encoding="1252")
    return df 


def save_df_as_table(df, table, oDmApp):

    temp_df = df.copy()
    path = rf"C:\temp\temp1.csv"

    if os.path.exists(path): os.remove(path)

    # change the column names
    temp_df.columns = [c.replace(" ", "_") for c in temp_df.columns]

    columns = temp_df.columns
    numeric_columns = temp_df.select_dtypes(include='number').columns
    object_columns = temp_df.columns.difference(numeric_columns)

    # cast non-numeric columns to str
    temp_df[object_columns] = temp_df[object_columns].astype(str)
    object_columns_length = {c: temp_df[c].str.len().max() for c in object_columns}

    temp_df.to_csv(path, index=None, header=None)

    columns_desc = "\n".join([
        f"'{c} N Y -'" if c in numeric_columns else f"'{c} A {object_columns_length[c]} Y -'"
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

    oDmApp.ParseCommand(command.replace("\n", ""))


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
