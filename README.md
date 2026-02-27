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

Install dependencies with the `pip install` command, and then add these to the requirements file by 
executing: 
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
add your api key as the `WANDB_API_KEY` environment variable, or use another method described 
[here](https://docs.wandb.ai/guides/integrations/add-wandb-to-any-library/#install-the-wandb-library-and-log-in).

Rename of `PROJECT_NAME` variable in [this file](src/common/wandb.py) to the project name you set up
on the weights and biases website and where you want the results and artifacts to be stored.

To enable logging from ultralytics, run this command in the cli:
```bash
yolo settings wandb=True
```

### Troubleshooting

#### `AttributeError: 'PosixPath' object has no attribute 'split'`
The first time after installing the wandb package you might run into some issues when training a 
yolov11 model, specifically.
The [issue](https://github.com/wandb/wandb/issues/10177#issuecomment-3102982507) has been reported 
with the developers on GitHub.

If there is still no fix for this in a more recent version of `wandb`, change this line 147 in 
[wandb/integration/ultralytics/callback.py](.venv/lib/python3.12/site-packages/wandb/integration/ultralytics/callback.py#L147) 
from:
```python
model.overrides["model"].split('.')[0]
```
to:
```python
f"{model.overrides["model"]}".split('.')[0]
```

#### Mismatch with GPU drivers and cuda libraries
Getting an error log like:
```
E0000 00:00:1764603865.710806   60625 cuda_dnn.cc:522] Loaded runtime CuDNN library: 9.1.0 but 
source was compiled with: 9.3.0. CuDNN library needs to have matching major version and equal or 
higher minor version. If using a binary install, upgrade your CuDNN library. If building from 
sources, make sure the library loaded at runtime is compatible with the version specified during 
compile configuration.
```

Pytorch has very strict requirements about the used cudu libraries, so it will complain when trying 
to upgrade the cuda libraries, which is necessary after upgrading your GPU drivers.
I have found that (at least for some cases) reinstalling tensorflow with: 
```
pip install tensorflow[and-cuda]
```
This upgrades the cuda libraries without the restrictions of pytorch, without breaking pytorch 
functionality, at least as for as is needed in this project.

# Structure

## root folder
Python files in the root folder are meant to be run as scripts from the command line with: 
```
python <file-name>.py
```

### .ipynb notebooks
Several Jupyter notebook files are included, these are meant as experimentation tools to try out 
code or run some specific one time operations with the different models.

#### playground.ipynb
Main experimentational notebook to build/debug/test source code.

#### hpe_poc.ipynb
Contains code for benchmarking the MediaPipe and YOLO hpe tools, and comparing and visualizing the
results. 

#### sota_vs_dnn.ipynb
Contains code for the model selection process (training and testing) for the SOTA image 
classification and DNN models.
Also compares and visualizes their results.
Additionally, a real world usecase is simulated to compare performance of the models that were found
best from both model selection processes.

#### dnn_vs_lstm.ipynb
Contains code for the model selection process (training and testing) for the RNN (LSTM) models.
Also compares and visualizes their results against those from the DNN models.

### CLI tools
The root folder also contains several cli tools for generating specific dataset.
Run these scripts with the `--help` flag to get tool specific information.

## /data
Contains all data files required in this project.
Label files will be included in source control to have back-ups of them, video files and images 
won't since they are too large and not mine to make publicly available.

### /aug
Examples of augmentation transformations are stored here, only for illustration purposes.

### /df
Will store various generated dataset as DataFrame objects.
The DataFrames are saved as `.pkl` files.

### /hpe
Contains data for the HPE benchmark. 
`/img` contains the open source images of climbers with the HPE landmarks labelled, these can be 
downloaded from [roboflow](https://app.roboflow.com/vaf/bouldering-poses/browse). 
The file structure under `/data/hpe/img` is determined by the downloaded files from Roboflow, 
however, as no training is done in this project with these images, they can all be moved to the 
`test` folder to have as much data as possible.

Benchmarking results are also placed in this folder, results for the MediaPipe tool are placed in 
`data/hpe/mediapipe` and for YOLO in `data/hpe/yolo`.
Comparison results of both tools are placed in `data/hpe`.

The file structure under `/data/hpe/mediapipe` and `/data/hpe/yolo` is determined by the source code 
of this project.

### /img
Contains the image datasets, generated from the annotated videos and used for training the SOTA 
models.
This structure is determined by the source code of this project.

### /labels
Labels for the video files in `/data/videos`. 
They should have the exact same names as the video files, but have the `.csv` extension.

#### /labels.yml
Configuration file to specify the labels the project will try to recognize and which are annotated
in the videos.

Expected structure: 
```yaml
name: "label name"
values:
  - INVALID
  - LABEL_1
  ...
```
The `name` value of the labels is also used as the default dataset name when these are generated.

`values` is a list of the names of the labels. 
These names are used as the folder names in the `/samples` and `/img` datasets. 
The indexes of these labels are used in the `.csv` label files under `/data/labels` and the `/df` 
datasets.

The only fixed value of these labels is the first label `INVALID`, at index `0`. 
This index will always be recognized as the `INVALID` label, used to mark video segments that should 
not be used for training or testing, for example, parts of the video that are heavily editted or do 
not contain data the models need to learn from.

### /runs
Contains the results from training, validation and test runs for the different models.

This structure is determined by the source code of this project, the actual saved files for each run 
depend on the used tools for that model.

The `split` folder are only present for models trained under the k-fold algorithm.

### /samples
Contains the video snippets grouped by their label. These samples are [generated](#3-extract-labelled-samples-from-videos) from `/videos` and `/labels`.

### /video
Folder that contains the full, original, videos that are used as dataset for this project. 

## /src
Contains the source code required in the project.

## /test
Contains tests for the source code, run with
```
pytest
```

# Usage

## 1. HPE tool benchmark
To compare the MediaPipe and Yolo tools for the HPE predictions the code in the 
[hpe poc notebook](/hpe_poc.ipynb) can be used, or the source code under `src/hpe`.

First, labelled images will be needed, we did the labelling with 
[roboflow](https://app.roboflow.com) and they can be downloaded 
[here](https://app.roboflow.com/vaf/bouldering-poses/browse). 

When using the project to analyze and classify (segments of) videos, this part can likely be 
skipped.

## 2. download videos
By running
```bash
python build-database.py
```
the videos used as data source will be downloaded and saved in the `/data/videos` folder.
First add links to the public videos you want to use for your project to the `__list` variable in 
[the python script](build-database.py).

These are just the public videos collected from the internet, your own videos can also be added 
by manually pasting them inside the output directory as well. 

## 3. label videos
By running
```bash
python play-video.py <video-path>
```
A video will be played while the current frame number is also shown in the player window for easier 
labelling of the videos.

The argument `video-path` passed to the script is the path to the video to play, this should be a 
video present in the `data/videos` directory.

Using the arrow keys you can walk through the video to find the relevant segment faster and more 
accurately.
While the video is playing, the arrows will jump 1 second in their respective direction.
While paused the video will jump only a single frame when pressing an arrow key.
Press `q` the quit the player, closing it with the mouse will just reopen it on the next frame.

Manually create a `.csv` file in the `/data/labels` directory with the same file name as the video 
your labelling.
These csv files should contain 4 columns:
Start Frame (inclusive) | End Frame (exclusive) | label value | cvs start
:---|:---:|:---:|---:
271|419|1|1
419|502|2|0
502|545|1|1
648|752|0|1

This would label the segment from frame 271 up to 419 as LABEL2 and as the start of a Continuous 
Valid Segment (CVS).

These label numbers start at 0 after the first INVALID label, they are indexes of only the valid 
labels returned by [iterate_valid_labels](src/labels.py).
All gaps between the end frame and the start frame from the next line are segments labelled as 
INVALID, indicating that part of the video cannot be used to train or test the models.

Transitions in labels can be over a continuous part of the video (first and second row), in this 
case end frame of the first and start frame of the next segment are the same and the second line
is not the start of a CVS.

Transitions in labels can with a cut to a different 'scene' (first and third row), in this 
case end frame of the first and start frame of the next segment are still the same, with the cut 
happening on this frame, but now the second line/segment is the start of a new CVS.

Two labelled segments can be separated by an invalid segment (third and fourth row).
Here the second row is always the start of a new CVS.

These CVS's are important to simulate the real world use cases of a real-time video classification 
tool, and to generate the datasets to train the RNN models.

## 4. Extract labelled samples from videos
With all videos new present and labelled, the valid segments can be extracted and stored separately 
as their own video fragments.
Do this by running 
```bash
python generate-sample-dataset.py
```
Each labelled part of the videos will be copied and pasted to the `/data/samples` directory.

This script expects videos directly in the `/data/videos` directory and files of the same name (but 
with the `.csv` extension) in the `/data/labels` directory.
It also expects a label configuration file at `/data/labels/labels.yml` with the names of the 
labels. 

Within the `/data/samples` directory, a folder will be made for each label according to their name 
in the [labels.yml file](/data/labels/labels.yml), except the INVALID label.

Each sample will be named according to the pattern:
```python
filename = f"{video_file_name}__{start_frame}.mp4"
```

## 5. HPE extraction from videos
In order to train the DNN and LSTM models the HPE features are first required.
These are extracted from the entire videos first, by running:
```bash
python extract-video-landmarks.py
```

Each video in `data/videos` is iterated over and frame-by-frame the hep landmarks are detected by 
the MediaPipe model.
These landmarks are stored per video in DataFrame objects with the naming pattern: 
```python
f"data/df/videos/{video_name}.pkl"
```

These dataset are prepended with a `frame_num` column so the frame of these landmarks can always be 
found, and a `label` column to have easier access to the annotation on that frame.

Run with the `--inspect` flag to view the extracted landmarks as they are generated.

Note: this might be improved/optimized by only extracting landmarks from the CVS's.

## 6. HPE segment extract and evaluation
Now the HPE landmarks of each individual labelled segment (video footage in `data/samples`) can be 
extracted and evaluated.
This is done separately for each label by running:
```bash
python extract-segment-landmarks.py <label-name> --interactive
```

This iterates over all segments for that label under `data/samples/<label-name>` and plays them, 
annotated with the corresponding HPE landmarks from `data/df/videos`.

Here the user has a chance to label these landmarks as accepted for the entire segment or not.
This information is stored in `data/df/segments/<label-name>/accepted.npy` as a numpy array.

The landmark features datasets are stored, per segment, with the naming strategy: 
```python
f"data/df/segments/{label-name}/{video_name}__{start_frame}.pkl"
```
So they correspond with the names of the segment video files in `data/samples/`.

Run without the `--interactive` flag to not show the videos annotated with the landmarks, and 
instead auto-accept the landmarks for all segments.

## 7. Build image dataset
From these labelled segments in `data/samples` the image datasets can now be generated, by running: 
```bash
python generate-image-dataset.py
```

This script expects video segments within the `/data/samples` directory, grouped in folders named 
after their true label.
So a video segment labelled as `LABEL1` should be found at: 
```
/data/samples/LABEL1/<file-name>.mp4
```

This dataset will be needed when training the SOTA image classification models that use single 
images as input or DNN models that used data extracted from these images as input.

This script will walk through each video in `/data/samples` for each label and extract a set of 
images to build this image dataset.
How these images are sampled is explained in the doc string of the `__generate_image_dataset` 
function in [this file](src/sampling/images.py).
All the sampled images will be divided in in either the training (train), validation (val) or 
testing (test) dataset, this will be done according to a 70/15/15 split.

At this stage, a dataset name will be added in the file system structure, this name is taken from 
the `name` field in the [label config file](/data/labels/labels.yml).
So the path for an image in the training set, labelled with `LABEL1`, will be found at:
```python
f"/data/img/{dataset_name}/train/LABEL1/{sample_name}__{frame_num}.png"
```

Once the image dataset have been generated, the image classification models should be able to be 
trained.

This script also accepts an `--accepted` flag, if this is the case, the image datasets will only be
built from accepted segments (`data/df/segments/<label-name>/accepted.npy`).
Accepted datasets are named:
```python
f"/data/img/{dataset_name}_full/train/LABEL1/{sample_name}__{frame_num}.png"
```

Note: In hindsight, these accepted image datasets have little use for training SOTA models, but they 
are required for generating HPE datasets from only accepted segments.
Removing this requirement and moving/copying the accepted option to the `generate-hpe-dnn-dataset`
script would be an improvement.

## 8. Build HPE DataFrame dataset
Once the image datasets are generated, the datasets with corresponding HPE landmark features can be 
created, by running: 
```bash
python generate-hpe-dnn-dataset.py <dataset-name>
```
The HPE features will be extracted from the HPE segment datasets in `data/df/segments` by looking at 
the images in the dataset `dataset-name` and copying over the corresponding row, effectively 
building an equivalent dataset as that of the images, but with HPE landmarks as input features.

This script expects an image dataset under `/data/img/{dataset-name}/`, first grouped by their split 
(`train`, `val` or `test`) and then grouped by their label name, and HPE segment data under 
`data/df/segments/`.

For each data split of the image dataset, an equivalent a DataFrame is constructed, with each image 
converted to a row of features. This will give us the following 3 files: 
```python
f"/data/df/{dataset-name}/train.pkl"
f"/data/df/{dataset-name}/val.pkl"
f"/data/df/{dataset-name}/test.pkl"
```

These dataset are used to train the Dense Neural Network (DNN) models.

## 9. Generate RNN HPE dataset precursor
To use the HPE landmark features to train RNN models a slightly different structure is needed.
This is generated by running:
```bash
python generate-rnn-dataset.py
```

This script iterates over all videos in `data/videos` and uses the label files in `data/labels` to 
find all Continuous Valid Segments (CVSs).
From all the CVSs it extracts the HPE landmarks from the files in `data/df/videos` and combines them
in one DataFrame object and stores it as `data/df/rnn/cvs_features.pkl`.

This dataset is prepended with three columns, `video` and `frame_num` so the frame of the HPE 
features can be found, and `group` which are integers that indicate which CVS the row belongs to.

## 10. Generate K-Fold datasets
All models and code examples below use K-fold cross validation to train models, generating these 
folds is done programatically during training.
So the image and hpe datasets (for dnn models), that were generated with splits, need to be combined
so the K-fold algorithms can generate these splits.

For the image datasets this is done manually by combining the images of different splits:
```python
f"/data/img/{dataset-name}/{split}/{label}/{image-name}.pkl"
```
in a single folder named: 
```python
f"/data/img/{dataset-name}_kf/all/{label}/{image-name}.pkl"
```

For HPE datasets for the DNN models this requires combining .pkl files, so this is done with the CLI
command:
 ```bash
python generate-hpe-dnn-dataset.py <dataset-name> --combine
```
Which will generate a new file:
```python
f"/data/df/{dataset-name}_kf/all.pkl"
```

The dataset to train RNN models is already combined in one file, so combining of splits is not 
needed for this dataset.

## 11. Train models
Individual models can be trained using this source code.
Depending on the type of model, different assumptions will be made from the `/data` folder.

### 11.1. SOTA image classification model
SOTA models will use the data in the `/data/img` folder, as it is generated by the 
`generate-image-dataset` script.
As SOTA models, [yolov11](https://docs.ultralytics.com/tasks/classify/) is used, and they can be 
used with the [SOTA class](src/sota/model.py).

For examples on how to train these models look at the `SOTA - Model Selection` part of 
[this notebook](sota_vs_dnn.ipynb)

### 11.2. HPE DNN model
The DNN models will use HPE features as input data, so they will use the `/data/df/<dataset-name>` 
folder. 
Multiple architectures for the DNN are defined [here](src/hpe_dnn/architecture.py) and denoted by 
the enum `DnnArch`.
Do not change these architectures when you have already used them, but add new ones if you want to 
try new network structures and compare them against the others.

To train these DNN models use code samples from the `HPE DNN - Model Selection` part of 
[this notebook](sota_vs_dnn.ipynb).

### 11.3 RNN model
Training RNN models will use the DataFrame object `data/df/rnn/cvs_features.pkl` as dataset.
Code samples to train these models can be found [here](dnn_vs_lstm.ipynb).

Similar to the DNN models, there are several architectures defined in 
[this file](src/rnn/architecture.py) and denoted by the enum `RnnArch`.

