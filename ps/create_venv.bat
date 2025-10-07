
REM check python version
python --version

REM crear el ambiente de python 
python -m venv .env

REM activar el ambiente de python
.env\Scripts\activate

REM install the libraries
python -m pip install -r requirements.txt
