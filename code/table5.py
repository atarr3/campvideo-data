import pandas as pd

from os.path import join
from sklearn.metrics import confusion_matrix

# root folder for replication repo
ROOT = '..'

# video directory
VID_DIR = join(ROOT, 'data', 'videos')

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# MTurk results
MT_DIR = join(ROOT, 'data', 'mturk')

# function for reading in WMP / CMAG data
def read_wmp():
    # open file
    wmp = pd.read_csv(join(WMP_DIR, 'wmp_final.csv'), index_col='creative')
    
    # sort index and return
    return wmp.sort_index()    

def main():
    # get WMP data
    wmp = read_wmp()
    
    # get MTurk results for identifying test set
    test_ids = pd.Index(
                pd.read_csv(join(MT_DIR, 'mood_mturk.csv')).creative.unique()
                ).sort_values()
    
    # subset down to videos with music
    mood_wmp = wmp.loc[wmp.music1 + wmp.music2 + wmp.music3 >= 1, 
                                    ['music1', 'music2', 'music3']].astype(int)
    # read in our results
    mood_pred = pd.read_csv(join(ROOT, 'data', 'mood_results.csv'),
                            index_col='creative')
    
    # results
    
    # wmp labels for test set
    mood_te = mood_wmp.loc[test_ids]
    
    # get predictions on test set
    m1_pred = mood_pred.loc[mood_te.index, 'music1']
    m2_pred = mood_pred.loc[mood_te.index, 'music2']
    m3_pred = mood_pred.loc[mood_te.index, 'music3']
    m4_pred = mood_pred.loc[mood_te.index, 'music4']
    
    # confusion matrix
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    cm1 = pd.DataFrame(
            confusion_matrix(mood_te.music1, m1_pred, normalize='all'),
            columns=cols, index=rows
            )
    cm2 = pd.DataFrame(
            confusion_matrix(mood_te.music2, m2_pred, normalize='all'),
            columns=cols, index=rows
            )
    cm3 = pd.DataFrame(
            confusion_matrix(mood_te.music3, m3_pred, normalize='all'),
            columns=cols, index=rows
            )
    cm4 = pd.DataFrame(
            confusion_matrix(mood_te.music1 | mood_te.music3, m4_pred, 
                             normalize='all'),
            columns=cols, index=rows
            )
    
    # save to txt file
    with open(join(ROOT, 'results', 'tables', 'table5.txt'), 'w') as fh:
        print("Music Mood Results (Ominous/Tense)", file=fh)
        print("----------------------------------", file=fh)
        print(cm1, file=fh)
        print(file=fh)
        print("Music Mood Results (Uplifting)", file=fh)
        print("------------------------------", file=fh)
        print(cm2, file=fh)
        print(file=fh)
        print("Music Mood Results (Sad/Sorrowful)", file=fh)
        print("----------------------------------", file=fh)
        print(cm3, file=fh)
        print(file=fh)
        print("Music Mood Results (Combined)", file=fh)
        print("-----------------------------", file=fh)
        print(cm4, file=fh)
        
if __name__ == '__main__':
    main()