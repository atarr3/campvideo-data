#!/bin/bash
set -ex

# preprocess WMP data
Rscript preprocess_CMAG.R

# generate summaries and audio features
python generate_data.py --no-gcp --overwrite 

# summary results
python summary_validation.py

# issue / opponent mention results
python --no-negativity text_validation.py

# face recognition results
python facerec_validation.py

# music mood results
python mood_validation.py

# ad negativity results
python --no-mention text_validation.py

# create tables

# create figures


mkdir -p ../results/unidirectional
python -u Unidirectional_LSTM.py
matlab -nodisplay -r "test_train_unidirectional"

mkdir -p ../results/bidirectional
python -u Bidirectional_LSTM.py
matlab -nodisplay -r "test_train_bidirectional"

mkdir -p ../results/biconcat
python -u Biconcat_LSTM.py
matlab -nodisplay -r "test_train_cascaded"