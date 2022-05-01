# Feature Extraction
To replicate the feature extraction step for creating all data in [``data/intermediate``](``data/intermediate``), follow the instructions below in order as they appear. Note that some instructions are duplicates from the [validation step](README.md). If these steps have already been performed, users may skip them.

## Table of Contents
1. [Data](#Data)
2. [Installation](#Installation)
3. [Feature Extraction](#Feature-Extraction)
4. [Additional Notes](#Additional-Notes)

## Data
Replication of the [Feature Extraction](#Feature-Extraction) step requires the collection of YouTube videos in MP4 format. Unfortunately, this dataset can be provided publicly. We provide a list of the YouTube Video IDs used in [``data/matches/matches.csv``](data/matches/matches.csv) under the `uid` variable. Users able to obtain these videos should place them in the [``data/videos``](data/videos) folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.

## Installation
Recreating all figures, tables and results requires working installations of
- [Python](https://www.python.org/downloads/), version 3.9 or greater

All code in this repo was tested under Python version 3.9.7 on a Windows 10 machine. 

### Prequisites
#### CMake and C++ Compiler
Installing the required Python packages requires both CMake and a C++ compiler. For macOS users, these requirements are normally already satisfied.
- C++ Compiler: Windows users should install a C++ compiler from [Microsoft Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/). Be sure to install the latest x86/x64 C++ build tools and the appropriate Windows SDK for your Windows platform. For example, a Windows 10 user would install
  - MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
  - Windows 10 SDK (Latest version)
- CMake: Install CMake via the command

  ```sh
  pip install cmake
  ```

### Google Cloud Platform (GCP)
Image text recognition and speech transcription are performed using GCP. Enabling GCP on your machine requires creating a project and setting up a billing account [here](https://cloud.google.com/docs/get-started). Once the account is setup, be sure to enable the following APIs:
- Google Cloud Vision API
- Google Cloud Video Intelligence API

**Note that using GCP costs money**. Setting up a GCP account and replicating this section will result in charges being made to your billing account.

### Python Dependencies
#### dlib
Windows users must build and install the ``dlib`` package from its [GitHub repository](https://github.com/davisking/dlib). After cloning the repository, navigate to the folder and enter

```sh
python setup.py install --no DLIB_GIF_SUPPORT
```

macOS users may skip this step.

#### Other packages
The remaining Python package dependencies can be installed by installing the project-related [``campvideo``](https://test.pypi.org/project/campvideo/) package. Both Windows and macOS users should install this package via

```sh
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple campvideo
```

### Models Download
After installing the ``campvideo`` package, download the relevant models via the command

    download_models
    
## Feature Extraction
The intermediate data in [``data/intermediate``](``data/intermediate``) can be replicated via

```sh
python scripts/generate_data.py --overwrite
```

The ``overwrite`` flag signals the script to replace existing data in [``data/intermediate``](``data/intermediate``). Without this flag, the script will skip over videos with existing data. If the user wishes to do partial replication of the feature extraction step **without** GCP, the command

```sh
python scripts/generate_data.py --overwrite --no-gcp
```

will compute audio features and video summaries only.

## Additional Notes
- Extracting features requires considerable computation. Expect this step to take several days to process every video.
- The GCP results rely on a stable internet connection. Service interruptions while executing this code may result in some files not being generated or overwritten.
