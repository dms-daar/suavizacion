
clear-host 

# C:\Users\Datamine\AppData\Local\miniconda3\shell\condabin\conda-hook.ps1
# conda activate 'C:\Users\Datamine\AppData\Local\miniconda3'
conda activate pyrmenv

set-location ./src

python -B -m pytest ../test -vv

set-location ..
