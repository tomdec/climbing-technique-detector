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

## FFMPEG
Make sure ffmpeg is installed for saving video's after editing. 
When using linux (or wsl) run
```
sudo apt update
sudo apt install ffmpeg
```

## Weights and biases
To integrate with a weights and biases server to log training, validation and evaluation of model, 
add your api key as the `WANDB_API_KEY` environment variable, or use another method described [here](https://docs.wandb.ai/guides/integrations/add-wandb-to-any-library/#install-the-wandb-library-and-log-in).

To enable logging from ultralytics, run this command in the cli:
```
yolo settings wandb=True
```

# Structure

## root folder
Python files in the root folder are meant to be run as scripts from the command line with: 
```
python <file-name>.py
```

## /data
Contains all data files required in this project.
Label files will be included source control to have back-ups of them, video files and images won't since they are too large and not mine
to make publically available.

## /src
Contains the source code required in the project.

### /src/notebook.ipynb
Jupyter notebook mean as experimentation tool to build/debug/test source code.

## /test
Contains tests for the source code, run by 
```
pytest
```