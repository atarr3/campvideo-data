import os
import pandas as pd

from itertools import product
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
    
    # get cleaned tonecmag variable
    tone = wmp.dropna(subset=['tonecmag'])
    # drop 'contrast' observations
    tone.drop(tone.loc[tone.tonecmag == 'CONTRAST'].index, inplace=True)
    # recast tonecmag to 1/0 for sentiment
    tone_wmp = ((tone.tonecmag == 'POS') | 
                (tone.tonecmag == 'POSITIVE')).astype(int)
    
    # read in our predictions
    neg_pred = pd.read_csv(join(ROOT, 'results', 'negativity_results.csv'),
                          index_col=['creative', 'feature', 'model', 'train'])
    
    ## results ##
    
    # confusion matrices
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    # negativity results
    model_name = {'lsvm': 'Linear SVM', 'nsvm': 'Non-linear SVM', 'knn': 'KNN',
                  'rf': 'Random Forest', 'nb': 'Naive Bayes'}
    feature_name = {'text': 'Text Only', 'music': 'Music Only', 
                    'both': 'Text + Music'}
    
    y_test = tone_wmp.loc[neg_pred.xs(0, level='train'
                                 ).index.get_level_values('creative'
                                 ).unique()]
    
    # delete old files
    if os.path.exists(join(ROOT, 'tables', 'tableS14-6.txt')):
        os.remove(join(ROOT, 'tables', 'tableS14-6.txt'))
    
    for model, feature in product(['lsvm', 'knn', 'rf', 'nb'], 
                                  ['text', 'music', 'both']):
        # predictions
        y_pred = neg_pred.xs((model, feature, 0), level=['model', 'feature', 'train'])
    
        # confusion matrix
        cm = pd.DataFrame(confusion_matrix(y_test, y_pred, normalize='all'),
                          columns=cols, index=rows
                         )
    
        # write results
        with open(join(ROOT, 'tables', 'tableS14-6.txt'), 'a') as fh:
            # model name
            if feature == 'text': 
                print("== "+ model_name[model] + " ==", file=fh)
            print(file=fh)
            print(feature_name[feature], file=fh)
            print('-' * len(feature_name[feature]), file=fh)
            print(file=fh)
            print(cm, file=fh)
            print(file=fh)
            # extra newline for final entry
            if feature == 'both':
                print(file=fh)

if __name__ == '__main__':
    main()