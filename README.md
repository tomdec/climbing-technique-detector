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
```bash
yolo settings wandb=True
```

### Troubleshooting

#### `AttributeError: 'PosixPath' object has no attribute 'split'`
The first time after installing the wandb package you might run into some issues when training a yolov11 model, specifically.
The [issue](https://github.com/wandb/wandb/issues/10177#issuecomment-3102982507) has been reported with the developers on GitHub.

If there is still no fix for this in a more recent version of `wandb`, change this line 147 in [wandb/integration/ultralytics/callback.py](.venv/lib/python3.12/site-packages/wandb/integration/ultralytics/callback.py#L147) from:
```python
model.overrides["model"].split('.')[0]
```
to:
```python
f"{model.overrides["model"]}".split('.')[0]
```

# Structure

## root folder
Python files in the root folder are meant to be run as scripts from the command line with: 
```
python <file-name>.py
```

### notebook.ipynb
Jupyter notebook mean as experimentation tool to build/debug/test source code.
Also contains code for training the models with all the used configurations.

## /data
Contains all data files required in this project.
Label files will be included source control to have back-ups of them, video files and images won't since they are too large and not mine
to make publicly available.

### /video
Folder that contains the full, original, videos that are used as dataset for this project. 

### /labels
Labels for the video files in `/data/videos`. They should have the exact same names as the video files, but have the `.csv` extension.

### /samples
### /img
### /df
### /runs

## /src
Contains the source code required in the project.

## /test
Contains tests for the source code, run by 
```
pytest
```

# Usage

## 1. download videos
By running
```bash
python build-database.py
```
the videos used as data source will be downloaded and saved in the `/data/videos` folder.
These are just the public video's we collected from the internet, your own videos can also be added by pasting them inside this directory as well. 

## 2. label videos
By running
```bash
python play-video.py <n>
```
the n'th video in the `/data/videos` directory will be played with the current frame number also displayed. 
Using the arrow keys you can walk through the video to find the relevant segment faster and more accurately.
While the video is playing, the arrows will jump 1 second in their respective direction.
While paused the video will jump only a single frame when pressing an arrow key.
Press `q` the quit the player, closing it with the mouse will just reopen it on the next frame.

Manually create a `.csv` file in the `/data/labels` directory with the same file name as the video your labelling.
These csv files should contain 3 columns:
Start Frame (inclusive) | End Frame (exclusive) | label index
:---|:---:|---:
271|419|1

This would label the segment from frame 271 up to 419 as label 1.

All gaps between the end frame and the start frame from the next line are segments labelled as label 0 (INVALID), indicating that part of the video cannot be used to train or test the models.

## 3. Extract labelled samples from videos
When all videos in `/data/videos` are labelled by `.csv` files in the `/data/labels` directory, the samples can be generated.
By running 
```bash
python generate-sample-dataset.py
```
Each labelled part of the videos will be copied and pasted to the `/data/samples` directory.
Within this directory, a folder will be made for each label according to their name in the enum [Technique](/src/labels.py), except `INVALID`.
Each sample will be named according to the pattern:
```python
filename = f"{video_file_name}__{start_frame}.mp4"
```

## 4. Building image dataset
When training models that use single images (or data from single images) as input, the image dataset will need to be created.
Do this by running: 
```bash
python generate-image-dataset.py
```
This script will walk through each video in `/data/samples` for each label and extract a set of images to build this image dataset.
How these images are sampled is explained in the doc string of the [this function](src/sampling/images.py#46).
All the sampled images will be divided in in either the train, validation or test dataset, this will be done according to a 70/15/15 split.

At this stage, the dataset name `techniques` will be added in the file system structure, so the path for an image in train set will be found at:
```python
f"/data/img/techniques/train/{label_name}/{sample_name}__{frame_num}.png"
```
