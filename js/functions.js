// -------- helpers (IE/JScript compatible) --------
function pad2(n){ return (n<10?'0':'') + n; }

function nowStamp(){
  var d = new Date();
  return d.getFullYear() + "" + pad2(d.getMonth()+1) + "" + pad2(d.getDate())
       + "_" + pad2(d.getHours()) + pad2(d.getMinutes()) + pad2(d.getSeconds());
}

function trimEndIE(str){
  return String(str).replace(/[\s\n\r]+$/, "");
}

function showError(msg){
  // Try alert (IE/Browser), then console, then WSH (HTA)
  try { alert(msg); return; } catch(e){}
  try { console.error(msg); return; } catch(e){}
  try { WScript.Echo(msg); } catch(e){}
}

function write_text_file(path, text){
  var file = null;
  try{
    // 2 = ForWriting, create if not exists
    file = FSO.OpenTextFile(path, 2, true, 0);
    file.Write(text);
  } finally {
    if(file) try{ file.Close(); }catch(e){}
  }
}

function get_file_contents(filePath){
  var file = null;
  if (!FSO.FileExists(filePath)) {
    throw Error("Error: file does not exist - " + filePath);
  }
  try{
    file = FSO.OpenTextFile(filePath, 1, false); // ForReading
    return file.AtEndOfStream ? "" : file.ReadAll();
  } catch(e){
    throw Error("Error: cannot read content - " + filePath);
  } finally {
    if(file) try{ file.Close(); }catch(_){}
  }
}

function delete_if_empty(path){
  try{
    if (FSO.FileExists(path)) {
      var txt = get_file_contents(path);
      if (trimEndIE(txt) === "") { FSO.DeleteFile(path, true); }
    }
  } catch(e) {
    // ignore cleanup problems
  }
}

function delete_file_if_exists(path){
  try{ if (FSO.FileExists(path)) { FSO.DeleteFile(path, true); } } catch(e){}
}

// -------- core runners --------
function run_cmd(command){
  // Always use fresh error file to avoid locking issues
  var stamp = nowStamp();
  var outPath   = TEMP_FOLDER + "\\OUTPUT.txt";
  var errPath   = TEMP_FOLDER + "\\ERROR_" + stamp + ".txt";

  // ensure previous OUTPUT.txt is gone
  delete_file_if_exists(outPath);

  // Build `cmd /c "<command> > "out" 2> "err""`
  // Quote redirection targets to survive spaces in TEMP_FOLDER
  var cmdline = [
    "cmd", "/c",
    command,
    ">",  '"' + outPath + '"',
    "2>", '"' + errPath + '"'
  ].join(" ");

  // Hidden window (0), wait=true
  var rc = SHELL.Run(cmdline, 0, true);

  var stdOut = "";
  var stdErr = "";
  try { if (FSO.FileExists(outPath)) stdOut = get_file_contents(outPath); } catch(e){}
  try { if (FSO.FileExists(errPath)) stdErr = get_file_contents(errPath); } catch(e){}

  // Error handling: non-zero exit OR any stderr content → surface & keep file
  if (rc !== 0 || trimEndIE(stdErr) !== "") {
    var msg = "Execution failed (code " + rc + ").\n\n" + stdErr + 
              "\n\nSaved at:\n" + errPath;
    showError(msg);
    throw Error(msg);
  }

  // No errors → remove the (empty) error file
  delete_if_empty(errPath);

  // Tidy OUT and return
  return trimEndIE(stdOut);
}

function run_python(script, args){
  // Build JSON argument (escape inner quotes for cmd)
  var json_args = JSON.stringify(args);
  json_args = '"' + json_args.split('"').join('""') + '"';

  var pyPath = '"' + SCRIPTS_FOLDER + "\\" + script + '"'; // quote script path

  // conda run -n <env> python "<script>" "<json>"
  var command = [
    "conda", "run", "-n", CONDA_ENV,
    "python", pyPath, json_args
  ].join(" ");

  var stdOut = run_cmd(command);

  // If your Python always prints a single JSON line, parse it
  // (If it may print logs + JSON, consider making Python write JSON to a file
  //  and read that file here instead.)
  var response = null;
  try{
    response = JSON.parse(stdOut);
  } catch(parseErr){
    // If parse fails, capture it as a proper error file + alert
    var msg = "Python returned non-JSON output.\n\nOutput:\n" + stdOut;
    showError(msg);
    throw Error(msg);
  }
  return response;
}
