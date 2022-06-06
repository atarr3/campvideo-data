import numpy as np
import pandas as pd

from os.path import join
from sklearn.metrics import confusion_matrix

# root folder for replication repo
ROOT = '..'

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# issue vocabulary list
VOCAB_PATH = join(ROOT, 'data', 'auxiliary', 'issuenames.csv')
VOCAB = pd.read_csv(VOCAB_PATH)

# function for reading in WMP / CMAG data
def read_wmp():
    # open file
    wmp = pd.read_csv(join(WMP_DIR, 'wmp_final.csv'), index_col='creative')
    
    # sort index and return
    return wmp.sort_index()

def main():  
    # read in WMP data
    wmp = read_wmp()
    
    # merge issue30 (abortion) and issue58 (women's health)
    wmp['issue30'] = np.logical_or(wmp.issue30, wmp.issue58).astype(int)
    # merge issue53 (healthcare) and issue59 (obamacare)
    # NOTE: issue59 only labeled in 2014 data
    wmp['issue53'] = np.logical_or(wmp.issue53, wmp.issue59, 
                                   out=wmp.issue53.to_numpy(),
                                   where=~wmp.issue59.isna())
    
    # subset to issue
    iss_wmp = wmp.loc[:, VOCAB.wmp.to_list()].astype(int, errors='ignore')
    
    # convert non-binary variables to binary (first 10 columns)
    iss_wmp.iloc[:, :10] = (~np.logical_or(iss_wmp.iloc[:, :10] == 0, 
                                           iss_wmp.iloc[:, :10] == 'No')).astype(int)
    
    # read in our predictions
    iss_pred = pd.read_csv(join(ROOT, 'results', 'mentions_results.csv'),
                          index_col=['creative', 'feature'])
    
    ## results ##
    
    # issues
    iss_text = iss_pred.xs('text', level='feature').drop(columns=['uid', 'o_mention'])
    iss_both = iss_pred.xs('both', level='feature').drop(columns=['uid', 'o_mention'])
    
    # confusion matrices
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    # issues
    cm_iss_text = pd.DataFrame(
                        confusion_matrix(iss_wmp.values.ravel(), 
                                         iss_text.values.ravel(), 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    cm_iss_both = pd.DataFrame(
                        confusion_matrix(iss_wmp.values.ravel(), 
                                         iss_both.values.ravel(), 
                                         normalize='all'),
                        columns=cols, index=rows
                        )

    # save results
    with open(join(ROOT, 'results', 'tables', 'table2.txt'), 'w') as fh:
        print("Issue Mention (Text Only)", file=fh)
        print("-------------------------", file=fh)
        print(cm_iss_text, file=fh)
        print(file=fh)
        print("Issue Mention (Text + Image)", file=fh)
        print("----------------------------", file=fh)
        print(cm_iss_both, file=fh)

if __name__ == '__main__':
    main()