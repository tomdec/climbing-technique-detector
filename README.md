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

#### Mismatch with GPU drivers and cuda libraries
Pytorch has very strict requirements about the used cudu libraries, so it will complain when trying to upgrade the cuda libraries, which is necessary after upgrading your GPU drivers.
I have found that (at least for some cases) reinstalling tensorflow with: 
```
pip install tensorflow[and-cuda]
```
This upgrades the cuda libraries without the restrictions of pytorch, without breaking pytorch functionality, at least as for as is needed in this project.

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
Label files will be included in source control to have back-ups of them, video files and images won't since they are too large and not mine
to make publicly available.

### /df
Contains the DataFrame datasets used for the HPE DNN models. The DataFrames are saved as `.pkl` files.

The file system structure of this folder is determined by the source code of this project.

Expected structure: 
```bash
data/df
├── {dataset_name}
│   ├── test.pkl
│   ├── train.pkl
│   └── val.pkl
└── {dataset_name}_kf
    └── all.pkl
```

### /hpe
Contains data for the HPE benchmark. These are open source images of climbers with the HPE landmarks labelled. Also contains the result data of the compared HPE tools, MediaPipe and Yolo.

The file structure under `/data/hpe/img` is determined by the HPE labelling tool, [roboflow](https://app.roboflow.com).

The file structure under `/data/hpe/mediapipe` and `/data/hpe/yolo` is determined by the source code of this project.

Expected structure:
```bash
data/hpe
├── img
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── test
│   │   ├── images
│   │   │   ├── image_name_1.jpg
│   │   │   └── ...
│   │   └── labels
│   │       ├── image_name_1.txt
│   │       └── ...
│   ├── train
│   │   ├── images
│   │   │   └── ...
│   │   └── labels
│   │       └── ...
│   └── valid
│       ├── images
│       │   └── ...
│       └── labels
│           └── ...
├── mediapipe
│   ├── avg-distances.png
│   ├── distances.pkl
│   └── distances_bgr.pkl
└── yolo
    ├── avg-distances.png
    ├── distances.pkl
    └── distances_bgr.pkl
```

### /img
Contains the image datasets. This structure is determined by the source code of this project.

Expected structure: 
```bash
data/img
├── {dataset_name}
│   ├── test
│   │   ├── label1
│   │   │   ├── image_name_1.png
│   │   │   └── ...
│   │   ├── ...
│   │   └── label{n}
│   │       ├── image_name_x.png
│   │       └── ...
│   ├── train
│   │   ├── label1
│   │   │   └── ...
│   │   └── ...
│   └── val
│       ├── label1
│       │   └── ...
│       └── ...
└── {dataset_name}_kf
    └── all
        ├── label1
        │   ├── image_name_1.png
        │   └── ...
        ├── ...
        └── label{n}
            ├── image_name_x.png
            └── ...
```

### /labels
Labels for the video files in `/data/videos`. They should have the exact same names as the video files, but have the `.csv` extension.

#### /labels.yml
Configuration file to specify the labels we will try to recognize.

Expected structure: 
```yaml
name: "label name"
values:
  - INVALID
  - LABEL_1
  ...
```
The `name` value of the labels is also used as the dataset name when these are generated.

`values` is a list of the names of the labels. These names are used as the folder names in the `/samples` and `/img` datasets. The indexes of these labels are used in the `.csv` label files under `/data/labels` and the `/df` datasets.

The only fixed value of these labels is the first label `INVALID`, at index `0`. This index will always be recognized as the `INVALID` label, used to mark video segments that should not be used for training or testing, for example, parts of the video that are heavily editted or cuts happen. 

### /runs
Contains the results from training, validation and test runs for the different models.

This structure is determined by the source code of this project, the actual saved files for each run depend on the used tools for that model.

The `split` folder are only present for models trained under the k-fold algorithm.

Expected structure:
```bash
data/runs
├── {model_name}
│   ├── train
│   │   └── ...
│   ├── ...
│   ├── train{n_train}
│   │   └── ...
│   ├── test
│   │   └── ...
│   ├── ...
│   └── test{n_test}
│       └── ...
└── {model_name}-kf-fold{n_fold}
    ├── split
    │   ├── test.npy
    │   ├── train.npy
    │   └── val.npy
    ├── train
    │   └── ...
    ├── ...
    ├── test
    │   └── ...
    └── ...
```

### /samples
Contains the video snippets grouped by their label. These samples are [generated](#3-extract-labelled-samples-from-videos) from `/videos` and `/labels`.

Expected structure:
```bash
data/samples
├── label1
│   ├── sample_name_1.mp4
│   └── ...
├── ...
└── label{n}
    ├── sample_name_x.mp4
    └── ...
```

### /video
Folder that contains the full, original, videos that are used as dataset for this project. 

## /src
Contains the source code required in the project.

## /test
Contains tests for the source code, run by 
```
pytest
```

# Usage

## 1. HPE tool benchmark
To compare the MediaPipe and Yolo tools for the HPE predictions the code in the [hpe poc notebook](/hpe_poc.ipynb) can be used, or the source code under `src/hpe`.

First, labelled images will be needed, we did the labelling with [roboflow](https://app.roboflow.com). 
Once labelled, the data can be imported and placed in `/data/hpe/img`.

Plot the distances between the labelled landmarks and the predicted landmarks, run:
```python
from src.hpe.yolo.performance import estimate_distances
from src.hpe.yolo.plot import plot_yolo_average_distances

distances = estimate_distances()
plot_yolo_average_distances(distances=distances)
```

To estimate the performance and log it to the output, run:
```python
from src.hpe.yolo.performance import estimate_performance

estimate_performance(name="Yolov11x-pose")
```

Similar code is used for MediaPipe, just import from `src.hpe.mp` instead.
Alterations can also be made to test the performance when evaluation the images from BGR color encoding, see the [hpe poc notebook](/hpe_poc.ipynb) for this code.

## 1. download videos
By running
```bash
python build-database.py
```
the videos used as data source will be downloaded and saved in the `/data/videos` folder.
These are just the public videos we collected from the internet, your own videos can also be added by pasting them inside this directory as well. 

## 2. label videos
By running
```bash
python play-video.py <n>
```
A video will be played while the current frame number is also shown in the player window for easier labelling of the videos.

The argument `n` passed to the script is an integer and indicates the index of the video to be played.

This script expects a number of videos directly in the `/data/videos` directory.

Using the arrow keys you can walk through the video to find the relevant segment faster and more accurately.
While the video is playing, the arrows will jump 1 second in their respective direction.
While paused the video will jump only a single frame when pressing an arrow key.
Press `q` the quit the player, closing it with the mouse will just reopen it on the next frame.

Manually create a `.csv` file in the `/data/labels` directory with the same file name as the video your labelling.
These csv files should contain 3 columns:
Start Frame (inclusive) | End Frame (exclusive) | label value
:---|:---:|---:
271|419|1

This would label the segment from frame 271 up to 419 as label 1.

All gaps between the end frame and the start frame from the next line are segments labelled as label 0 (INVALID), indicating that part of the video cannot be used to train or test the models.

## 3. Extract labelled samples from videos
By running 
```bash
python generate-sample-dataset.py
```
Each labelled part of the videos will be copied and pasted to the `/data/samples` directory.

This script expects videos directly in the `/data/videos` directory and files of the same name (but with the `.csv` extension) in the `/data/labels` directory.
It also expects a label configuration file at `/data/labels/labels.yml` with the names of the labels 

Within the `/data/samples` directory, a folder will be made for each label according to their name in the [labels.yml file](/data/labels/labels.yml), except the label with value `0`.

Each sample will be named according to the pattern:
```python
filename = f"{video_file_name}__{start_frame}.mp4"
```

## 4. Build image dataset
By running: 
```bash
python generate-image-dataset.py
```
an image dataset will be generated from the labelled video samples.

This script expects video segments within the `/data/samples` directory, grouped in folder named after their true label.
So a video segment labelled as `Y1` should be found at: 
```
/data/samples/Y1/<file-name>.mp4
```

This dataset will be needed when training models that use single images (or data from single images) as input.

This script will walk through each video in `/data/samples` for each label and extract a set of images to build this image dataset.
How these images are sampled is explained in the doc string of the [this function](src/sampling/images.py#34).
All the sampled images will be divided in in either the training (train), validation (val) or testing (test) dataset, this will be done according to a 70/15/15 split.

At this stage, a dataset name will be added in the file system structure, this name is taken from the `name` field in the [label config file](/data/labels/labels.yml). 
So the path for an image in the training set, labelled with `Y1`, will be found at:
```python
f"/data/img/{dataset_name}/train/Y1/{sample_name}__{frame_num}.png"
```

## 5. Build DataFrame dataset
By running: 
```bash
python generate-hpe-dnn-dataset.py
```
HPE features will be calculated and stored in a `.pkl` file for working with a Dense Neural Network (DNN) model.

This script expects an image dataset under `/data/img/{dataset_name}/`, first grouped by their split (`train`, `val` or `test`) and then grouped by their label name.

Each image found there will be processed by an HPE model and a specific set of features will be extracted, these features can be found [here](src/hpe/landmarks.py#36) for the pose landmarks and [here](src/hpe/landmarks.py#58) for the hand landmarks.

For each data split a DataFrame is then constructed, with each image converted to a row of features. This will give us the following 3 files: 
```
/data/df/{dataset_name}/train.pkl
/data/df/{dataset_name}/val.pkl
/data/df/{dataset_name}/test.pkl
```

## 6. Train models
Individual models can be trained using this source code.
Depending on the type of model, different assumptions will be made from the `/data` folder.

### 6.1. SOTA image classification model
SOTA models will use the data in the `/data/img` folder. This code will expect the following file structure there:
```python
f"/data/img/{dataset_name}/{split}/{label_name}/{file_name}.png"
```
As SOTA models, [yolov11](https://docs.ultralytics.com/tasks/classify/) models are used, and they can be used with the [SOTA class](src/sota/model.py#51).
For example to train this model execute: 
```python
from src.common.model import ModelInitializeArgs
from src.sota.model import SOTA, SOTAConstructorArgs, SOTATrainArgs

sota = SOTA(args=SOTAConstructorArgs(name="name"))
sota.initialize_model(args=ModelInitializeArgs())
sota.train_model(args=SOTATrainArgs(epochs=10))
``` 

### 6.2. HPE DNN model
The DNN models will use HPE features as input data, so they will use the `/data/df` folder. There the code expects data to be organized as
```python
f"/data/df/{dataset_name}/{split}.pkl"
```
Multiple architectures for the DNN are defined [here](src/hpe_dnn/architecture.py) and denoted by the enum `DnnArch`.
Do not change these architectures when you have already used them, but add new ones if you want to compare them against the others.

To train these DNN models execute: 
```python
from src.hpe_dnn.architecture import DnnArch
from src.hpe_dnn.model import HpeDnn, HpeDnnConstructorArgs, HpeDnnModelInitializeArgs, HpeDnnTrainArgs  

hpednn = HpeDnn(args=HpeDnnConstructorArgs(
    name="name", 
    model_arch=DnnArch.ARCH1))
hpednn.initialize_model(args=HpeDnnModelInitializeArgs())
hpednn.train_model(args=HpeDnnTrainArgs(epochs=10))
```

## 7. Combine data splits for K-Fold Validation
These model architectures can also be compared according to the K-Fold Cross Validation algorithm.
But this requires there to be a single dataset with all data, not one that has already been split in a train, val and test set.

### 7.1. SOTA image classification model
For the `/data/img` dataset, this is easiest done manually by copy and paste'ing the images from the dataset that has the split.
The expected folder structure here would be: 
```python
f"/data/img/{dataset_name}_kf/all/{label_name}/{image_name}.png
```

### 7.2. HPE DNN model
For the data in `/data/df` this is a bit more complex as we need to combine data from multiple `.pkl` files.
This can be done by running the code: 
```bash
python generate-hpe-dnn-dataset.py -c
```
This will expect these three dataset splits to already exist
```
/data/df/{dataset_name}/train.pkl
/data/df/{dataset_name}/val.pkl
/data/df/{dataset_name}/test.pkl
```
and combine them to
```
/data/df/techniques_kf/all.pkl
```

## 8. K-Fold validation of models
Multiple models can be trained in a group according to the K-Fold Cross Validation Algorithm using this source code.
Depending on the type of model, different assumptions will be made from the `/data` folder.
All data will be split in 10 different folds.

### 8.1. SOTA image classification model
The SOTA models can be trained with K-Fold Cross Validation to acquired multiple test metrics.
This code will assume the existence of the combined dataset:
```python
f"/data/img/{dataset_name}_kf/all/{label_name}/{image_name}.png
```

To run the K-Fold Cross Validation, run: 
```python
from src.common.model import ModelConstructorArgs
from src.sota.model import SOTAModelInitializeArgs, SOTAMultiRunTrainArgs, SOTATrainArgs
from src.sota.kfold import SOTAFoldCrossValidation

cross_validation = SOTAFoldCrossValidation(
    model_args = ModelConstructorArgs(name="name-kf"),
    train_run_args = SOTAMultiRunTrainArgs(
        runs=5,
        model_initialize_args=SOTAModelInitializeArgs(model="yolo11m-cls"),
        train_args=SOTATrainArgs(epochs=10)
    )
)

cross_validation.train_folds()
```

### 8.2. HPE DNN model
The DNN models can also be trained with K-Fold Cross Validation.
This code will assume the existence of the dataset:
```python
f"/data/df/{dataset_name}_kf/all.pkl"
```

To run the K-Fold algorithm, execute:
```python
from src.hpe_dnn.architecture import DnnArch
from src.hpe_dnn.model import HpeDnnModelInitializeArgs, HpeDnnMultiRunTrainArgs, HpeDnnTrainArgs
from src.common.model import ModelConstructorArgs
from src.hpe_dnn.kfold import HpeDnnFoldCrossValidation

cross_validation = HpeDnnFoldCrossValidation(
    model_args=ModelConstructorArgs(name="name-kf"),
    train_run_args=HpeDnnMultiRunTrainArgs(
        runs=5,
        model_initialize_args=HpeDnnModelInitializeArgs(model=DnnArch.ARCH1),
        train_args=HpeDnnTrainArgs(epochs=10)
    ))

cross_validation.train_folds()
```