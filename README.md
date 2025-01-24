# climbing-technique-detector

# Requirements

Python version: a minimum version of python 3.9 is required. 
In this application we used [Python 3.12.3](https://www.python.org/downloads/release/python-3123/)

To use this codebase, one must have all the necessary requirements. 
We provide all libraries which are used in this project as part of the ```requirements.txt``` file. 
We recommend using a virtual environment during the setup. Do this by running: 
```
python3 -m venv ./.venv
```

Activate the virtual environment by running:
On Windows:
```
.venv\Scripts\Activate.ps1
```
On linux: 
```
source .venv/bin/activate
```

To properly set up the project run the following command from the root of this project:

```
pip install -r requirements.txt
```

Install dependencies with the `pip install` command, and then add these to the requirements file by executing: 
```
pip freeze > requirements.txt
```