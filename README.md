# Python optimization

This repo has code which demonstrates some python performance problems and potential solutions


## Installation
Create a python virtual environment

### Windows powershell
```
python3.9 -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux/MacOS bash
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

For other platforms google 'create python virtual environment on _platform_'

## Building pybind11 module
```
cd code
pip install .
```
This compiles the pybind11 module on your computer, and installs the resulting shared library
so that it is available in your python environment

