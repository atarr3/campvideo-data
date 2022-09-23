#!/bin/bash
set -ex

# preprocess WMP data
Rscript preprocess_CMAG.R

# generate summaries and audio features
python generate_data.py --no-gcp --overwrite 

# summary results
python summary_validation.py

# issue / opponent mention and ad negativity results
python text_validation.py

# face recognition results
python facerec_validation.py

# music mood results
python mood_validation.py

# create tables
printf "Creating tables... "

Rscript table1.R

python table2.py

python table3.py

python table4.py

python table5.py

python table6.py

Rscript tableS1-1.R

python tableS14-5.py

python tableS14-6.py

Rscript tableS14-7.R

Rscript tableS14-8.R

printf "Done!\n"

# create figures
printf "Creating figures... "

Rscript figure5.R

Rscript figure8_S14-9_S14-10.R

python figureS7-4.py

python figureS13-8.py

printf "Done!\n"
