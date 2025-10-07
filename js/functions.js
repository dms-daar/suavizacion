

function get_file_contents(filePath) {

        if (!FSO.FileExists(filePath)) {
          throw Error("Error: File does not exist - " + filePath);
        }


          try {

            file = FSO.OpenTextFile(filePath, 1, false);

            // var output = []
            // while (! file.AtEndOfStream) {
            //   output.push(file.ReadLine)
            // }
            // output = output.join('\n')

            if (!file.AtEndOfStream) {
              var output = file.ReadAll();
            } else {
              var output = "";
            };
            
            

          } catch (e) {

            throw Error("Error: cannot read content - " + filePath);

          } finally {
              file.Close();
          }
          return output;
      }


function run_cmd(command) {

    var method = 1

    if(method == 1) {

    function trimEndIE(str) {
        // Replace spaces and newline characters (\s covers spaces, tabs, and newlines) at the end of the string.
        return str.replace(/[\s\n]+$/, '');
    }

        var command_with_redirections = [
        "cmd", 
        "/c",
        command, 
        '>', TEMP_FOLDER + '/OUTPUT.txt', 
        '2>', TEMP_FOLDER + '/ERROR.txt',
        ].join(" ");

        var oRun = SHELL.Run(command_with_redirections, 0, true); 

        if(oRun != 0) {
            var stdErr = get_file_contents(TEMP_FOLDER + '/ERROR.txt');
            throw Error('\n\n\n\n\n\n\n\n\n\n\n\n' + stdErr);
        }

        var stdOut =  get_file_contents(TEMP_FOLDER + '/OUTPUT.txt');
        stdOut = trimEndIE(stdOut); // stdOut.slice(0, stdOut.length-3)
        return stdOut; 

    } else {

    var command = [
        "cmd", 
        "/c",
        command, 
    ].join(" ");

                // run the command
    var exe = SHELL.exec(command);
    // wait for the end of the execution
    while (exe.Status == 0) {
        1+1;
    }
    // read the standard error
    var stdOut = exe.stdOut.ReadAll();
    var stdErr = exe.StdErr.ReadAll();

    // return the error level and standard output
    if (stdErr != "") {throw Error(stdErr);}

    // return the standard output        
    return stdOut.slice(0, stdOut.length-2); 

    }

    }


function run_python(script, args) {
    // build the command
    var json_args = JSON.stringify(args);
    var json_args = '"' + json_args.split('"').join('""') + '"';

    var command = [
    "conda", 
    "run", 
    "-n", 
    CONDA_ENV, 
    "python", 
    SCRIPTS_FOLDER + "\\" + script, 
    json_args, 
    ].join(" ");

    var stdOut = run_cmd(command);
    var response = JSON.parse(stdOut);
    return response; 

}

