import pandas as pd

from os.path import abspath, dirname, join
from pkg_resources import resource_filename
from sklearn.metrics import confusion_matrix

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# issue vocabulary list
VOCAB_PATH = resource_filename('campvideo','data/issuenames.csv')
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
    
    # subset to o_mention column
    oment_wmp = wmp.loc[:, 'o_mention'].dropna().astype(int)
    
    # read in our predictions
    iss_pred = pd.read_csv(join(ROOT, 'results', 'mentions_results.csv'),
                          index_col=['creative', 'feature'])
    
    ## results ##
    
    # opponent mentions
    oment_text = iss_pred.xs('text', level='feature').o_mention
    oment_both = iss_pred.xs('both', level='feature').o_mention
    
    # confusion matrices
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    # opponent mentions
    cm_oment_text = pd.DataFrame(
                        confusion_matrix(oment_wmp, 
                                         oment_text.loc[oment_wmp.index], 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    cm_oment_both = pd.DataFrame(
                        confusion_matrix(oment_wmp, 
                                         oment_both.loc[oment_wmp.index], 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    with open(join(ROOT, 'tables', 'table3.txt'), 'w') as fh:
        print("Opponent Mention (Text Only)", file=fh)
        print("----------------------------", file=fh)
        print(cm_oment_text, file=fh)
        print(file=fh)
        print("Opponent Mention (Text + Image)", file=fh)
        print("-------------------------------", file=fh)
        print(cm_oment_both, file=fh)
 
if __name__ == '__main__':
    main()