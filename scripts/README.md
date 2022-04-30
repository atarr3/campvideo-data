# Feature Extraction
To replicate the feature extraction step for creating all data in ``data\intermediate``, follow the instructions below in order as they appear. Note that some instructions in [Installation](#Installation) are duplicates from the [Validation](../README.md#Validation) step. If these steps have already been performed, users may skip them.

## Data
Replication of the [Feature Extraction](#Feature-Extraction) step requires the collection of YouTube videos in MP4 format. Unfortunately, this dataset can be provided publicly. We provide a list of the YouTube Video IDs used in [``data/matches/matches.csv``](../data/matches/matches.csv) under the `uid` variable. Users able to obtain these videos should place them in the [``data/videos``](../data/video) folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.

## Installation
Recreating the intermediate results in the [Feature Extraction](#Feature-Extraction) step requires a working installations of [Python](https://www.python.org/downloads/) verion 3.9 or greater. All code in this repo was tested under Python version 3.9.7 on a Windows 10 machine.

### CMake and C++ Compiler
Installing the required Python packages requires both CMake and a C++ compiler. For macOS users, these requirements are normally already satisfied. Windows users should install a C++ compiler from [Microsoft Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/). Be sure to install the latest x86/x64 C++ build tools and the appropriate Windows SDK for your Windows platform.

CMake can be installed via the command

```sh
pip install cmake
```

### CUDA and cuDNN
We **strongly recommended** that users with access to a dedicated GPU for computing install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). Without GPU support, results will differ for face recognition, and performance will be much slower.

### Google Cloud Platform (GCP)
Image text recognition and speech transcription are performed using GCP. Enabling GCP on your machine requires creating a project and setting up a billing account [here](https://cloud.google.com/docs/get-started). Once the account is setup, be sure to enable the following APIs:
- Google Cloud Vision API
- Google Cloud Video Intelligence API

**Note that using GCP costs money**. Setting up a GCP account and replicating this section will result in charges being made to your billing account.

### Python Dependencies
#### dlib
Windows users must build the ``dlib`` package from its [GitHub repository](https://github.com/davisking/dlib). After cloning the repository, navigate to the folder and enter

```sh
python setup.py install --no DLIB_GIF_SUPPORT
```

macOS users may skip this step and proceed to the step below.

#### Other Packages
The remaining Python package dependencies can be installed by installing the project-related ``campvideo`` package, which is available on the [TestPyPi package repository](https://test.pypi.org/project/campvideo/). Both Windows and macOS users should install this package via

```sh
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple campvideo
```

### Model Download
After installing the ``campvideo`` package, download the relevant models via the command

    download_models
    
## Feature Extraction
The intermediate data in ``data\intermediate`` can be replicated via

    python scripts/generate_data.py --overwrite
    
The ``overwrite`` flag signals the script to replace existing data in ``data\intermediate``. Without this flag, the script will skip over videos with existing data. If the user wishes to do partial replication of the feature extraction step **without** GCP, the command

    python scripts/generate_data.py --overwrite --no-gcp
    
will compute audio features and video features only.
