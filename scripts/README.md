# Feature Extraction
To replicate the feature extraction step for creating all data in ``data\intermediate``, follow the instructions below in order as they appear.

## Data
Replication of the [Feature Extraction](#Feature-Extraction) step requires the collection of YouTube videos in MP4 format. Unfortunately, this dataset can be provided publicly. We provide a list of the YouTube Video IDs used in [``data/matches/matches.csv``](../data/matches/matches.csv) under the `uid` variable. Users able to obtain these videos should place them in the [``data/videos``](../data/video) folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.

## Installation
Recreating the intermediate results in the [Feature Extraction](#Feature-Extraction) step requires a working installations of [Python](https://www.python.org/downloads/) verion 3.9 or greater. All code in this repo was tested under Python version 3.9.7 on a Windows 10 machine.

### CUDA and cuDNN
We **strongly recommended** that users with access to a dedicated GPU for computing install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). Without GPU support, results will differ for face recognition, and performance will be much slower.

### Google Cloud Platform (GCP)
Image text recognition and speech transcription are performed using GCP. Enabling GCP on your machine requires creating a project and setting up a billing account [here](https://cloud.google.com/docs/get-started). Once the account is setup, be sure to enable the following APIs:
- Google Cloud Vision API
- Google Cloud Video Intelligence API

**Note that using GCP costs money**. Setting up a GCP account and replicating this section will result in charges being made to your billing account.

### Python Dependencies
#### dlib
The Python package ``dlib`` must be compiled from source in order to use CUDA and cuDNN. Windows users should follow the instructions [here](../readme.md#dlib). macOS users may skip to the next step below.

#### Other Packages
All other Python package dependencies can be installed by installing the project-related package, ``campvideo``, which is available on [TestPyPi package repository](https://test.pypi.org/project/campvideo/). This package can be installed within a Python environment via the command

    pip install -i https://test.pypi.org/simple/ campvideo

## Model Download
After installing the ``campvideo`` package, download the relevant models via the command

    download_models
    
## Feature Extraction
The intermediate data in ``data\intermediate`` can be replicated via

    python scripts/generate_data.py --overwrite
    
The ``overwrite`` flag signals the script to replace existing data in ``data\intermediate``. Without this flag, the script will skip over videos with existing data. If the user wishes to do partial replication of the feature extraction step **without** GCP, the command

    python scripts/generate_data.py --overwrite --no-gcp
    
will compute audio features and video features only.
