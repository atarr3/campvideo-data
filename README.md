# campvideo-data
Replication data for ["Automated Coding of Political Campaign Advertisement Videos: An Empirical Validation Study"]() by Alexander Tarr, June Hwang, and Kosuke Imai.

## Overview
Full replication of the results in the paper is a laborious process, involving significant setup and computation time on the part of the user. To simplify the procedure, we have split the replication process into two parts: [feature extraction](##-Feature-Extraction) and [validation](##-Validation). For those seeking only to validate the results in the paper, it is highly recommended to ignore the feature extraction step and follow the steps for validation, which makes use of already-extracted features.

## Repository Layout
This repository is split into several folders: ``data``, ``figs``, ``results``, ``scripts`` and ``tables``.
- ``data``: This folder contains all data needed to perform both feature extraction and validation.
  * ``ids``: Numpy vectors for face encodings corresponding to Senate candidates in the 2012 and 2014 elections.
  * ``intermediate``: Extracted feature data for each YouTube video in the study. This data includes pre-computed audio features, keyframe indices, auto-generated transcripts, and detected image text. Data in this folder is created in the [feature extraction](##-Feature-Extraction) step.
  * ``mturk``: CSV files containing data for the Amazon Mechanical Turk studies.
  * ``validation``: CSV files containing data for the validation analyses given in the appendix.
  * ``videos``: MP4 files corresponding to YouTube videos used in the study. Data in this folder is used in the [feature extraction](##-Feature-Extraction) step.
  * ``wmp``: DTA files containing WMP/CMAG data. Data in this folder is used in the [validation](##-Validation) step.
- ``figs``: PDFs for figures generated by the code that are displayed in the paper.
- ``results``: CSV files containing predicted labels for tasks studied in the paper. There are also raw text files showing general statistics about the performance of our methods that are discussed in the main text of the paper.
- ``scripts``: All code needed to generate data, extract features, validate results, and create figures and tables.
-  ``tables``: Raw text files showing confusion matrices corresponding to tables in the paper.

## Validation

## Feature Extraction

## Additional Notes
