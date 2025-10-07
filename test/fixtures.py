

import pytest 
from rmprocs.dm import * 


@pytest.fixture(
    scope='session', 
    autouse=True, 
    # params=[
    #     {"app": "StudioUG", "export_method": "EXTRA", "temp_folder": r"C:\temp"},
    #     {"app": "StudioUG", "export_method": "DmFileLib"},
    # ], 
    # ids=[
    #     "oDmApp(EXTRA)", 
    #     "oDmApp(DmFileLib)"    
    # ]
)
# def oDmApp(request):
def oDmApp():

    # before runnig the first test
    oDmApp, _ = get_oDmApp("StudioRM")

    yield oDmApp

    # after running the last test
    pass

