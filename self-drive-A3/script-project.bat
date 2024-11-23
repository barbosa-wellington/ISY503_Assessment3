@echo off
SET ENV_NAME=drive-car
SET PYTHON_VERSION=3.9

REM Create a new Conda environment with the specified Python version
echo Creating a new Conda environment: %ENV_NAME% with Python %PYTHON_VERSION%
conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y

REM Activate the newly created Conda environment
echo Activating the environment: %ENV_NAME%
call conda activate %ENV_NAME%

REM Install dependencies from the requirements.txt file in the project folder
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

REM Ensure the simulator is open manually before proceeding

REM Run the drive-car.py script
echo Running the drive-car.py script...
python drive-car.py

echo Script execution finished.
pause
