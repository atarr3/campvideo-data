import pandas as pd

from os.path import abspath, join, dirname
from sklearn.metrics import confusion_matrix

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

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
    
    # subset down to videos with music
    mood_wmp = wmp.loc[wmp.music1 + wmp.music2 + wmp.music3 >= 1, 
                                    ['music1', 'music2', 'music3']].astype(int)
    # MTurk data
    mood_mturk = pd.read_csv(join(MT_DIR, 'mood_mturk.csv'), index_col='creative')
    
    # majority prediction by MTurk
    major = mood_mturk.groupby(axis=0, level='creative'
                     ).apply(pd.DataFrame.mode
                     ).reset_index(0
                     ).set_index('creative')
    
    # confusion matrix
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('MTurk', 'No'), ('MTurk', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    cm1 = pd.DataFrame(
            confusion_matrix(mood_wmp.loc[major.index, 'music1'], 
                             major.music1_mturk, normalize='all'),
            columns=cols, index=rows
            )
    cm2 = pd.DataFrame(
            confusion_matrix(mood_wmp.loc[major.index, 'music2'], 
                             major.music2_mturk, normalize='all'),
            columns=cols, index=rows
            )
    cm3 = pd.DataFrame(
            confusion_matrix(mood_wmp.loc[major.index, 'music3'], 
                             major.music3_mturk, normalize='all'),
            columns=cols, index=rows
            )
    
    # MTurk confusion matrices
    with open(join(ROOT, 'tables', 'tableS14-1.txt'), 'w') as fh:
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

if __name__ == '__main__':
    main()