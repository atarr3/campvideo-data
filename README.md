# campvideo-data
Replication data for ["Automated Coding of Political Campaign Advertisement Videos: An Empirical Validation Study"]() by Alexander Tarr, June Hwang, and Kosuke Imai.

## Overview
Full replication of the results in the paper is a laborious process, involving significant setup and computation time on the part of the user. To simplify the procedure, we have split replication into two parts: [Feature Extraction](README-FE.md#Feature-Extraction) and [Validation](#Validation). For those seeking only to validate the results in the paper, it is **highly recommended** to ignore feature extraction and follow the steps for validation, which uses pre-computed features from the feature extraction step.

We provide instructions for replicating the [Validation](#Validation) step in this document, while instructions for replicating feature extraction are found in [README-FE.md](README-FE.md).

## Repository Layout
This repository is split into several folders: ``data``, ``figs``, ``results``, ``scripts`` and ``tables``.
- ``data``: This folder contains all data needed to perform both feature extraction and validation.
  * ``ids``: Numpy vectors for face encodings corresponding to Senate candidates in the 2012 and 2014 elections.
  * ``intermediate``: Extracted feature data for each YouTube video in the study. This data includes Numpy vectors for audio features, keyframe indices, auto-generated transcripts, and detected image text. Data in this folder is created in the [Feature Extraction](#Feature-Extraction) step.
  * ``matches``: CSV and JSON files containing information about matches between CMAG videos and YouTube videos. These files are used for computing coverage tables and preprocessing the raw WMP data.
  * ``mturk``: CSV files containing results from the Amazon Mechanical Turk studies.
  * ``validation``: CSV files containing results from the validation analyses discussed in the appendix.
  * ``videos``: MP4 files corresponding to YouTube videos used in the study. Data in this folder is used in the [Feature Extraction](#Feature-Extraction) step.
  * ``wmp``: DTA files containing WMP/CMAG data. Data in this folder is used in the [Validation](#Validation) step.
- ``figs``: PDFs for figures generated by the code that are displayed in the paper.
- ``results``: CSV files containing predicted labels for tasks studied in the paper. There are also raw text files showing general statistics about the performance of our methods that are discussed in the main text of the paper.
- ``scripts``: All code needed to generate data, extract features, validate results, and create figures and tables.
- ``tables``: Raw text files showing confusion matrices and coverage tables corresponding to tables in the paper.

## Data
Replication in the [Feature Extraction](#Feature-Extraction) step requires the human-coded labels provided by WMP. Unfortunately, we cannot share this dataset publicly. The WMP data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential, 2012 Non-Presidential, and 2014 data. The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the [``data/wmp``](data/wmp) folder.

## Installation
Recreating all figures, tables and results requires working installations of [Python](https://www.python.org/downloads/) version 3.9 or greater and [R](https://cran.r-project.org/src/base/R-4/). All code in this repo was tested under Python version 3.9.7 and R version 4.0.5 on a Windows 10 machine. 

### Prequisites
#### CMake and C++ Compiler
Installing the required Python packages requires both CMake and a C++ compiler. For macOS users, these requirements are normally already satisfied. Windows users should install a C++ compiler from [Microsoft Visual Studio Community Edition](https://visualstudio.microsoft.com/downloads/). Be sure to install the latest x86/x64 C++ build tools and the appropriate Windows SDK for your Windows platform.

CMake can be installed via the command

```sh
pip install cmake
```

#### CUDA and cuDNN
We **strongly recommended** that users with access to a dedicated GPU for computing install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). Without GPU support, results in [``results``](results), and performance will be much slower.

### Python Dependencies
#### dlib
Windows users must build and install the ``dlib`` package from its [GitHub repository](https://github.com/davisking/dlib). After cloning the repository, navigate to the folder and enter

```sh
python setup.py install --no DLIB_GIF_SUPPORT
```

macOS users may skip this step.

#### Other packages
The remaining Python package dependencies can be installed by installing the project-related ``campvideo`` package, which is available on the [TestPyPi package repository](https://test.pypi.org/project/campvideo/). Both Windows and macOS users should install this package via

```sh
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple campvideo
```

### R Dependencies
All R code uses the following packages: ``dplyr, here, lme4, quanteda, quanteda.sentiment, readstata13, readtext, stargazer, xtable``, most of which can be installed from within the R environment via

```r
install.packages("<PACKAGE_NAME>")
```

``quanteda.sentiment`` is not available on CRAN and must be installed via

```r
devtools::install_github("quanteda/quanteda.sentiment")
```

### spaCy Model Download
The ``spacy`` text modeling package requires downloading a model. After installing the Python packages, enter the following in the command line:

```sh
python -m spacy download en_core_web_md
```    
    
## Preprocessing the WMP Data
Before any results can be produced, the WMP data must be cleaned. After placing the Stata files into ``data\wmp``, clean the data via

```sh
Rscript scripts/preprocess_CMAG.R
```

This file may also be sourced from within an IDE, such as RStudio. Be sure to set the working directory to repo folder, ``campvideo-data``. After running, a file called ``wmp_final.csv`` should be created in [``data/wmp``](data/wmp).

## Result Replication
The following commands recreate the tables and figures in the paper. The generated figures are found in the ``figs`` folder, while the tables are stored in raw text files in the ``tables`` folder. Additionally, performance metrics discussed in the paper as well as our predicted labels are stored in the ``results`` folder.

### Coverage Tables
This section gives instructions for replicating the coverage tables (Section 2.2, Appendix S1).
- Table 1 in the main text is replicated via

```sh
Rscript scripts/table1.R
```

- Table S1.1 in the appendix is replicated via

```sh
Rscript scripts/tableS1-1.R
```

### Text Validation
This section gives instructions for replicating issue mention (Section 4.1, Appendix S11), opponent mention (Section 4.2, Appendix S12), and ad negativity classification (Section 4.5, Appendix S14.2, Appendix S14.3) results.
- Table 2, Table 3, Table 6, and Table S14.2 are replicated via

      python scripts/text_validation.py
  
  Note that this script uses pre-computed results in the ``results`` folder to construct the tables. To recreate the data in ``results``, type the command
  
      python scripts/text_validation.py --calculate
      
  The ``calculate`` flag forces the script to scan the auto-generated transcipts for issue and opponent mentions and to retrain the text models described in the paper using the WMP data as ground truth. The resulting predictions are then saved to the ``results`` folder.
  
- Figure 5 is replicated via

      Rscript scripts/fig5.R
      
- Performance metrics for issue mentions and opponent mentions are found in ``results\issue_results.txt`` and ``results\oppment_results.txt``, which are replicated with

      python scripts/text_validation.py

### Face Recognition Validation
This section gives instructions for replicating face recognition results (Section 4.3, Appendix S13).
- Table 4 and Figure S13.8 are replicated via

      python scripts/facerec_validation.py
      
  Note that this script uses pre-computed results in the ``results`` folder to construct the tables and figures. To recreate the data in ``results``, type the command
  
      python scripts/facerec_validation.py --calculate
      
  The ``calculate`` flag forces the script to detect and recognize faces in the keyframes of each video and to recompute the distance threshold. The resulting predictions are then saved to the ``results`` folder.
  
- Performance metrics for face recognition are found in ``results\facerec_results.txt``, which are replicated with

      python scripts/text_validation.py

### Music Mood Validation
This section gives instructions for replicating music mood classificaiton results (Section 4.4, Appendix S14.1).

- Table 5, and Table S14.5 are replicated via

      python scripts/mood_validation.py
      
  Note that this script uses pre-computed results in the ``results`` folder to construct the tables. To recreate the data in ``results``, type the command
  
      python scripts/mood_validation.py --calculate
      
  The ``calculate`` flag forces the script to retrain the music mood models described in the paper using the WMP data as ground truth.. The resulting predictions are then saved to the ``results`` folder.
      
- Figure 8, Figure S14.9, and Figure S14.10 are replicated via

      Rscript scripts/figs8_14-9_14-10.R
      
- Performance metrics for music mood classification are found in ``results\mood_results.txt``, which are replicated with

      python scripts/mood_validation.py

### Video Summary Validation
This section gives instructions for replicating results in the summary validation study (Appendix S7).

- Figure S7.4 is replicated via

      python scripts/summary_validation.py
      
  Note that this script uses pre-computed results in the ``results`` folder to construct the figure. To recreate the data in ``results``, type the command
  
      python scripts/summary_validation.py --calculate
      
  The ``calculate`` flag forces the script to compute all relevants metrics for each video summary. The results are then saved to the ``results`` folder.
  
### Ad Negativity Classification with LSD
This section gives instructions for replicating ad negativity classification results using LSD (Appendix S14.3).

- Table S14.7 is replicated via

      Rscript scripts/tableS14-7.R

### Kaplan *et al.* (2006) Replication
This section gives instructions for replicating the issue convergence study using our predicted labels (Appendix S14.4).

- Table S14.7 is replicated via
      
      Rscript scripts/tableS14-7.R

## Additional Notes
- Feature extraction, model training, and prediction require significant processing time. Expect full replication of the results in the paper to take several days. Conversely, recreating all figures and tables using pre-computed results and features takes very little time.
- Image text recognition and speech transcription with GCP require a stable internet connection. Service interruptions during execution of ``scripts/generate_data.py`` may lead to missing data.
- Exact replication for label prediction is only guaranteed for the models we train. Face recognition, image text recognition, and speech transcription all rely on external models which we have no control over. Future updates to these models may lead to slightly different results.
- 'File not found' errors are likely due to issues with working directory. All code assumes this repo, `campvideo-data`, is the working directory.
