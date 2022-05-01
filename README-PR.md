# Prediction
To replicate the prediction step for creating the data in [``results``](``results``), follow the instructions below in order as they appear. Note that some instructions are duplicates from the [Feature Extraction](README-FE.md#Installation) step. If these steps have already been performed, users may skip them.

## Table of Contents
1. [Data](#Data)
2. [Installation](#Installation)
3. [Results Replication](#Results-Replication)
4. [Additional Notes](#Additional-Notes)

## Data
Full replication of the [Prediction](#Prediction) step requires the human-coded labels provided by WMP and the YouTube videos for the ads in MP4 format. Unfortunately, we cannot share either of this data publicly.
- The WMP data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential, 2012 Non-Presidential, and 2014 data. The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the [``data/wmp``](data/wmp) folder.
- We provide a list of the YouTube Video IDs used in [``data/matches/matches.csv``](data/matches/matches.csv) under the `uid` variable. Users able to obtain these videos should place them in the [``data/videos``](data/videos) folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.

## Installation

### Prequisites
#### FFmpeg
FFmpeg is used for audio/video processing applications, such as video resizing and trimming. Users can download and install FFmpeg [here](https://ffmpeg.org/download.html) or through a package manager, such as Homebrew or APT. Once installed, confirm that FFmpeg is working by typing

```sh
ffpmeg
```

in terminal or command prompt.

#### CMake and C++ Compiler
Installing the required Python packages requires both CMake and a C++ compiler. For macOS users, these requirements are normally already satisfied.
- C++ Compiler: Windows users should install a C++ compiler from [Microsoft Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/). Be sure to install the latest x86/x64 C++ build tools and the appropriate Windows SDK for your Windows platform. For example, a Windows 10 user would install
  - MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
  - Windows 10 SDK (Latest version)
- CMake: Install CMake via the command

  ```sh
  pip install cmake
  ```

#### CUDA and cuDNN
We **highly recommended** that users with access to a dedicated GPU for computing install 
- [CUDA](https://docs.nvidia.com/cuda/index.html#installation-guides)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) 
 
Without GPU support, results in [``results``](results) will differ, and performance will be much slower.

### Python Dependencies
#### dlib
Windows users must build and install the ``dlib`` package from its [GitHub repository](https://github.com/davisking/dlib). After cloning the repository, navigate to the folder and enter

```sh
python setup.py install --no DLIB_GIF_SUPPORT
```

macOS users may skip this step. Note that installing ``dlib`` can be difficult, especially for Windows users. If the above command fails, try building ``dlib`` without GPU support via

```sh
python setup.py install --no DLIB_GIF_SUPPORT --no DLIB_USE_CUDA
```
Installing ``dlib`` without GPU support will still allow for exact replication of the figures and tables using pre-computed results, however the pre-computed results in [``results``](results) cannot be replicated.

#### Other packages
The remaining Python package dependencies can be installed by installing the project-related [``campvideo``](https://test.pypi.org/project/campvideo/) package. Both Windows and macOS users should install this package via

```sh
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple campvideo
```

### spaCy Model Download
The ``spacy`` text modeling package requires downloading a model. After installing the Python packages, enter the following command:

```sh
python -m spacy download en_core_web_md
```
## Results Replication
The replication code for the figures and tables relies on pre-computed results in the [``results``](results) folder. The CSV files in this folder contain the predicted labels and some feature information. The following table describes the different results files, the associated classification tasks, the script for generating the results file, and the Figures and Tables which depend on those results.

| Results File                                               | Classification Task          | Script                                                     |  Figure and Table Dependencies                              |
| :--------------------------------------------------------- | :--------------------------- | :--------------------------------------------------------- | :---------------------------------------------------------- |
| [``summary_results.csv``](results/summary_results.csv)     | Video Summarization          | [``summary_validation.py``](scripts/summary_validation.py) | Figure S7.4                                                 |
| [``mentions_results.csv``](results/mentions_results.csv)   | Issue/Opponent Mentions      | [``text_validation.py``](scripts/text_validation.py)       | Figure 5, Table 2, Table 3                                  |
| [``facerec_results.csv``](results/facerec_results.csv)     | Face Recognition             | [``facerec_validation.py``](scripts/facerec_validation.py) | Figure S13.8, Table 4                                       |
| [``mood_results.csv``](results/mood_results.csv)           | Music Mood Classification    | [``mood_validation.py``](scripts/mood_validation.py)       | Figure 8, Figure S14.9, Figure S14.10, Table 5, Table S14.1 |
| [``negativity_results.csv``](results/mentions_results.csv) | Ad Negativity Classification | [``text_validation.py``](scripts/text_validation.py)       | Table 6, Table S14.3, Table S14.6                           | 

These scripts can be executed via

```
python scripts/<SCRIPT>
```

where ``<SCRIPT>`` is given by the name in the "Script" column in the table above.

In addition to the CSV files, these scripts also produces raw text files containing various performance metrics reported in the main text. Like the figures and tables, these files rely on data in the CSV files. These files can be recreated without overwriting the CSV files via

```
python scripts/<SCRIPT> --no-calculate
```

where ``<SCRIPT>`` is given by the name in the "Script" column in the table above.

## Additional Notes
- Face recognition results will differ substantially if CUDA and cuDNN are not installed. This is due to the ``face_recognition`` package using differen face detection models in these scenarios. 
- Recreating the results CSV files is much more time-consuming due to extensive model training and file I/O. Expect this step to take upwards of a day.
- Exact replication for label prediction is only guaranteed for the models we train. Face recognition, image text recognition, and speech transcription all rely on external models which we have no control over. Future updates to these models may lead to slightly different results than those given in the paper.
