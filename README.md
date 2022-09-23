# campvideo-data
Replication data for "Automated Coding of Political Campaign Advertisement Videos: An Empirical Validation Study" by Alexander Tarr, June Hwang, and Kosuke Imai.

See [``description.txt``](description.txt) for a full explanation of all files and folders in this repository.

## Data
The replication code requires the human-coded labels provided by WMP and YouTube videos, both of which cannot be shared publicly. 
- The WMP data can be purchased [here](https://mediaproject.wesleyan.edu/dataaccess/). Our study used the 2012 Presidential (Version 1.2), 2012 Non-Presidential (Version 1.1), and 2014 data (Version 1.0). The data is distributed across 7 Stata files, one for each year and race type (House, Senate, Governor, President). These files should be placed in the [``data/wmp``](data/wmp) folder.
- The list of YouTube videos used in this study is found in the [``data/auxiliary/metadata.csv``](data/auxiliary/metadata.csv) file under the ``uid`` column, which denotes the YouTube Video ID. These files should be obtained in .mp4 format and placed in the [``data/youtube``](data/youtube) folder and named according to the YouTube Video ID (e.g., ``Pzepu1vdv78.mp4``).

## Environment
All code was tested under [Python 3.8](https://www.python.org/downloads/) and [R 4.1.3](https://cran.r-project.org/bin/) on the following platforms:
- Windows 10
- Windows 11
- Ubuntu 20.04

Note that hardware and OS differences may lead to small discrepancies from some of the results presented in this repository, which was run on Ubuntu 20.04.

### Dependencies
The list of package dependences and corresponding version used to produce the results are given below.

#### General

| Package   | Version |
| :-------- | :------ |
| libblas   | Latest  |
| cmake     | 3.16.3  |
| cuda      | 11.7    |
| cudnn     | 8.4.1   |
| ffmpeg    | 4.2.7   |
| liblapack | Latest  |

With the exception of CUDA and cuDNN, these packages can be installed with a package manager, such as ``apt-get``. CUDA and cuDNN can be installed following the instructions for your platform [here](https://docs.nvidia.com/cuda/) and [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), respectively.

#### R

| Package            | Version |
| :------------------| :------ |
| devtools           | 4.2.3   |
| dplyr              | 1.0.9   |
| lme4               | 1.1_29  |
| quanteda           | 3.2.1   |
| quanteda.sentiment | N/A     |
| readstata13        | 0.10.0  |
| stargazer          | 5.2.3   |
| xtable             | 1.8_4   |

Most of these packages can be installed from within the R environment via

```r
install.packages("PACKAGE_NAME")
```

``quanteda.sentiment`` is not available on CRAN and must be installed via

```r
devtools::install_github("quanteda/quanteda.sentiment")
```

#### Python

| Package          | Version  |
| :--------------- | :------- |
| campvideo        | 1.2.3    |
| dlib             | 19.24.0  |
| face_recognition | 1.3.0    |
| ffmpeg-python    | 0.2.0    |
| matplotlib       | 3.5.2    |
| numpy            | 1.22.4   |
| opencv-python    | 4.5.5.64 |
| pandas           | 1.4.2    |
| scikit-learn     | 1.0.1    |
| scipy            | 1.8.1    |
| seaborn          | 0.11.2   |
| spacy            | 3.3.0    |

These packages are installed via

```bash
pip install PACKAGE_NAME
```

The `campvideo` package must be installed via

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple campvideo
```

### Post-install
After downloading and installing all packages, download the spaCy text model via

```bash
python -m spacy download en_core_web_md
```

## Figure and Table Replication
Full replication is achieved through the [``run.sh``](code/run.sh) script, which generates all intermediate files and creates all tables and figures in the [``results``](results) folder. This script also computes any statistics reported in the text of our paper, which can be found in the [``results/performance``](results/performance) folder. Note that all code assumes [``code``](code) is the current working directory.

The full list of figures and tables and associated replication code is given below.

| Result        | Description                                                | Code                                                      |
| :------------ | :--------------------------------------------------------- | :-------------------------------------------------------- |
| Figure 5      | MTurk results for issue mentions                           | [``figure5.R``](code/figure5.R)                           |
| Figure 8      | MTurk results for ominous/tense mood classification        | [``figure8_S14-9_S14-10.R``](code/figure8_S14-9_S14-10.R) |
| Figure S7.4   | Video summarization validation study results               | [``figureS7-4.py``](code/figureS7-4.py)                   |
| Figure S13.8  | ROC plots for face recognition                             | [``figureS13-8.py``](code/figureS13-8.py)                 |
| Figure S14.9  | MTurk results for uplifting mood classification            | [``figure8_S14-9_S14-10.R``](code/figure8_S14-9_S14-10.R) |
| Figure S14.10 | MTurk results for sad/sorrowful mood classification        | [``figure8_S14-9_S14-10.R``](code/figure8_S14-9_S14-10.R) |
| Table 1       | Matched video coverage table                               | [``table1.R``](code/table1.R)                             |
| Table 2       | Confusion matrices for issue mentions                      | [``table2.py``](code/table2.py)                           |
| Table 3       | Confusion matrices for opponent mentions                   | [``table3.py``](code/table3.py)                           |
| Table 4       | Confusion matrices for face recognition                    | [``table4.py``](code/table4.py)                           |
| Table 5       | Confusion matrices for mood classiification                | [``table5.py``](code/table5.py)                           |
| Table 6       | Confusion matrices for ad negativity classification (NSVM) | [``table6.py``](code/table6.py)                           |
| Table S1.1    | YouTube channel coverage table                             | [``tableS1-1.R``](code/tableS1-1.R)                       |
| Table S14.5   | Confusion matrix for mood MTurk results                    | [``tableS14-5.py``](code/tableS14-5.py)                   |
| Table S14.6   | Confusion matrices for ad negativity classification (All)  | [``tableS14-6.py``](code/tableS14-6.py)                   |
| Table S14.7   | Confusion matrix for LSD results                           | [``tableS14-7.R``](code/tableS14-7.R)                     |
| Table S14.8   | Regression coefficients for issue convergence study        | [``tableS14-8.R``](code/tableS14-8.R)                     |
