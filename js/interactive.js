function BrowseDM(FileType){

    var oBrowser = oDmApp.ActiveProject.Browser
    oBrowser.TypeFilter = FileType
    oBrowser.Show(false);
    return(oBrowser.FileName);
    
}


function BrowseAndSave(FileType, elementId) {

    var file = BrowseDM(FileType);
    var elem = document.getElementById(elementId)
    elem.value = file
    return file
}


function columnPicklist(table, selectId) {

    var elem = document.getElementById(selectId);
    oScript.makeFieldsPicklist(table, elem)
}
