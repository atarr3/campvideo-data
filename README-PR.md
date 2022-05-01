# Prediction

## Data
Full replication of the [Prediction](#Prediction) step requires the human-coded labels provided by WMP and the YouTube videos for the ads in MP4 format. Unfortunately, we cannot share either of this data publicly.
- The WMP data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential, 2012 Non-Presidential, and 2014 data. The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the [``data/wmp``](data/wmp) folder.
- We provide a list of the YouTube Video IDs used in [``data/matches/matches.csv``](data/matches/matches.csv) under the `uid` variable. Users able to obtain these videos should place them in the [``data/videos``](data/videos) folder, with each video file titled ``<YouTubeID>.mp4``. ``<YouTubeID>`` is the unique YouTube video ID.

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
We **strongly recommended** that users with access to a dedicated GPU for computing install 
- [CUDA](https://docs.nvidia.com/cuda/index.html#installation-guides)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). 
 
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
Installing ``dlib`` without GPU support will still allow for exact replication of the figures and tables using pre-computed results, however the pre-computed results cannot be replicated.

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
