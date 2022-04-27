import json
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from os.path import abspath, join, dirname
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# video directory
VID_DIR = join(ROOT, 'data', 'videos')

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# MTurk results
MT_DIR = join(ROOT, 'data', 'mturk')

# lookup table mapping YouTube IDs to CMAG IDs
with open(join(ROOT, 'data', 'matches', 'matches.json'), 'r') as fh:
    MATCHES = json.load(fh)
    
# lookup table mapping CMAG IDs to YouTube IDs
MATCHES_CMAG = dict(zip(MATCHES.values(), MATCHES.keys()))

# function for reading in WMP / CMAG data and creating a single dataframe
def read_wmp():
    # check for WMP / CMAG data
    fpaths = [join(WMP_DIR, fname) for fname in os.listdir(WMP_DIR)]
    
    # throw error if no files found
    if len(fpaths) == 0:
        raise Exception('no WMP files found in directory `%s`' % WMP_DIR)
        
    # read in all files and combine into a single data frame
    res = pd.DataFrame()
    for fpath in fpaths:
        if fpath.endswith('.dta'):
            cur_data = pd.read_stata(fpath)
        elif fpath.endswith('.csv'):
            cur_data = pd.read_csv(fpath)
        else:
            raise Exception('WMP files must be in .csv or .dta format')
            
        # determine year from `airdate` variable and add to cur_data
        year = pd.DatetimeIndex(cur_data.airdate).year.max()
        cur_data.insert(cur_data.columns.get_loc('race'), 'year', year)
        
        # concatenate and remove duplicates (keep first)
        res = pd.concat([res, cur_data], ignore_index=True, sort=False
                        ).drop_duplicates(subset='creative')
        
        # subset down to matches
        matched = res.loc[res.creative.isin(MATCHES.values()), :]
        matched = matched.set_index('creative')
        matched = matched.sort_index()
            
    return matched
        

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-c', '--calculate', dest='calculate', 
                        action='store_true', default=False, help='Flag for '
                        'specifying whether or not to fit models and predict '
                        'labels. If not specified, the script will look for '
                        'results in `results/mood_results.csv`.')
    
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_arguments()    
    calculate = args.calculate
        
    # feature dimension
    fdim = 452
    
    # get WMP data
    wmp = read_wmp()
    
    # get MTurk results for identifying test set
    test_ids = pd.Index(
                pd.read_csv(join(MT_DIR, 'mood_mturk.csv')).creative.unique()
                ).sort_values()
    
    # subset down to videos with music
    mood_wmp = wmp.loc[wmp.music1 + wmp.music2 + wmp.music3 >= 1, 
                                    ['music1', 'music2', 'music3']].astype(int)
        
    # features
    if calculate:
        print('Reading in features...')
        feat = pd.DataFrame(index=mood_wmp.index, columns=np.arange(fdim), dtype=float)
        for creative in feat.index:
            # YT ID
            uid = MATCHES_CMAG[creative]
            # read in feature
            feat_path = join(ROOT, 'data', 'intermediate', uid, 'audiofeat.npy')
            with open(feat_path, 'rb') as fh:
                feat.loc[creative] = np.load(fh)
        print("Done!")
            
        # training and testing features
        feat_tr = feat.loc[~mood_wmp.index.isin(test_ids)]
        feat_te = feat.loc[test_ids]
            
        # training and testing data labels
        mood_tr = mood_wmp.loc[~mood_wmp.index.isin(test_ids)]
        mood_te = mood_wmp.loc[test_ids]
        
        # classification pipelines
        pipe1 = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', SVC(kernel='rbf', class_weight='balanced'))
                          ])
        pipe2 = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', SVC(kernel='rbf', class_weight='balanced'))
                          ])
        pipe3 = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', SVC(kernel='rbf', class_weight='balanced'))
                          ])
        pipe4 = Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', SVC(kernel='rbf', class_weight='balanced'))
                          ])
        
        # grid search for tuning parameters, optimizing accuracy or balanced accuracy
        params = [{'clf__gamma': np.logspace(-5, -2, 16),
                   'clf__C': np.geomspace(0.1, 10, 16)}]
        
        # fit classifiers
        print("Training classifiers...")
        clf1 = GridSearchCV(pipe1, params, scoring='accuracy', cv=5)
        clf2 = GridSearchCV(pipe2, params, scoring='accuracy', cv=5)
        clf3 = GridSearchCV(pipe3, params, scoring='balanced_accuracy', cv=5)
        clf4 = GridSearchCV(pipe4, params, scoring='accuracy', cv=5)
        
        print("    Training classifier 1...", end='\r', flush=True)
        clf1.fit(feat_tr, mood_tr.music1)
        print("    Training classifier 2...", end='\r', flush=True)
        clf2.fit(feat_tr, mood_tr.music2)
        print("    Training classifier 3...", end='\r', flush=True)
        clf3.fit(feat_tr, mood_tr.music3)
        print("    Training classifier 4...", end='\n', flush=True)
        clf4.fit(feat_tr, mood_tr.music1 | mood_tr.music3)
        print("Done!")
        
        # save results
        uids = [MATCHES_CMAG[ele] for ele in feat.index]
        mood_pred = pd.DataFrame({'train': 1, 'uid': uids, 
                                  'music1': clf1.predict(feat), 
                                  'music2': clf2.predict(feat), 
                                  'music3': clf3.predict(feat), 
                                  'music4': clf4.predict(feat)},
                                 index=feat.index)
        mood_pred.loc[feat_te.index, 'train'] = 0
        
        mood_pred.sort_index().to_csv(join(ROOT, 'results', 'mood_results.csv'))
        
    else:
        mood_pred = pd.read_csv(join(ROOT, 'results', 'mood_results.csv'),
                                index_col='creative')
    
    # results
    print("Summarizing results...")
    
    # MTurk data
    mood_mturk = pd.read_csv(join(MT_DIR, 'mood_mturk.csv'), index_col='creative')
    
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
    with open(join(ROOT, 'tables', 'table5.txt'), 'w') as fh:
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
    
    # statistics
    n_music = mood_wmp.shape[0]
    n_multi = (mood_wmp.music1 + mood_wmp.music2 + mood_wmp.music3 >= 2).sum()
    n_m1, n_m2, n_m3 = mood_wmp.music1.sum(), mood_wmp.music2.sum(), mood_wmp.music3.sum()
    
    # majority prediction by MTurk
    major = mood_mturk.groupby(axis=0, level='creative'
                     ).apply(pd.DataFrame.mode
                     ).reset_index(0
                     ).set_index('creative')
    
    # majority prediction agreement with WMP
    agree1 = mood_wmp.loc[major.index, 'music1'] == major.music1_mturk
    agree2 = mood_wmp.loc[major.index, 'music2'] == major.music2_mturk
    agree3 = mood_wmp.loc[major.index, 'music3'] == major.music3_mturk
    
    agree = (agree1.sum() + agree2.sum() + agree3.sum()) / (3 * agree1.shape[0])
    
    # confusion matrices
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
    
    # statistics
    with open(join(ROOT, 'results', 'mood_results.txt'), 'w') as fh:
        print("Music Mood Results", file=fh)
        print("------------------", file=fh)
        print("# of ads containing music: {} ({:.0%})".format(n_music, n_music/wmp.shape[0]), file=fh)
        print("# of ads w/ multiple labels: {} ({:.0%})".format(n_multi, n_multi/n_music), file=fh)
        print(file=fh)
        print("# of ads labeled Ominous/Tense: {} ({:.0%})".format(n_m1, n_m1/n_music), file=fh)
        print("# of ads labeled Uplifting: {} ({:.0%})".format(n_m2, n_m2/n_music), file=fh)
        print("# of ads labeled Sad/Sorrowful: {} ({:.0%})".format(n_m3, n_m3/n_music), file=fh)
        print(file=fh)
        print("Music Mood MTurk Results", file=fh)
        print("------------------------", file=fh)
        print("Majority of MTurk coders agree with WMP in {:.0%} of all cases".format(agree), file=fh)
    print("Done!")

if __name__ == '__main__':
    main()