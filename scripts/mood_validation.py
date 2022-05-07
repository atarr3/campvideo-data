import json
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from os.path import abspath, join, dirname
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

# feature data directory
FEAT_DIR = join(ROOT, 'data', 'features')

# lookup table mapping YouTube IDs to CMAG IDs
with open(join(ROOT, 'data', 'matches', 'matches.json'), 'r') as fh:
    MATCHES = json.load(fh)
    
# lookup table mapping CMAG IDs to YouTube IDs
MATCHES_CMAG = dict(zip(MATCHES.values(), MATCHES.keys()))

# function for reading in WMP / CMAG data
def read_wmp():
    # open file
    wmp = pd.read_csv(join(WMP_DIR, 'wmp_final.csv'), index_col='creative')
    
    # sort index and return
    return wmp.sort_index()      

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-nc', '--no-calculate', dest='calculate', 
                        action='store_false', default=True, help='Flag for '
                        'specifying to not fit models and predict labels. If '
                        'specified, the script will look for results in '
                        '`results/mood_results.csv`.')
    
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_arguments()    
    calculate = args.calculate
        
    # feature dimension
    d = 452
    
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
        print('Reading in features... ', end='', flush=True)
        # read in features, subset to wmp sample
        colnames = ['v' + str(ele) for ele in range(d)]
        feat = pd.read_csv(join(FEAT_DIR, 'features.csv'), index_col=['creative'], 
                           usecols=['creative'] + colnames).loc[mood_wmp.index]
        print("Done!")
            
        # training and testing features
        feat_tr = feat.loc[~mood_wmp.index.isin(test_ids)]
        feat_te = feat.loc[test_ids]
            
        # training data labels
        mood_tr = mood_wmp.loc[~mood_wmp.index.isin(test_ids)]
        
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
        mood_pred = pd.DataFrame({'uid': uids, 'train': 1, 
                                  'music1': clf1.predict(feat), 
                                  'music2': clf2.predict(feat), 
                                  'music3': clf3.predict(feat), 
                                  'music4': clf4.predict(feat)},
                                 index=feat.index)
        mood_pred.loc[feat_te.index, 'train'] = 0
        
        mood_pred.sort_index().to_csv(join(ROOT, 'results', 'mood_results.csv'))
        
        # update MTurk file with predictions
        mood_mturk = pd.read_csv(join(MT_DIR, 'mood_mturk.csv'), 
                                 index_col=['creative'])
        mood_mturk['music1_pred'] = mood_pred.loc[mood_mturk.index].music1
        mood_mturk['music2_pred'] = mood_pred.loc[mood_mturk.index].music2
        mood_mturk['music3_pred'] = mood_pred.loc[mood_mturk.index].music3
        mood_mturk['music4_pred'] = mood_pred.loc[mood_mturk.index].music4
        
        # save MTurk
        mood_mturk.to_csv(join(MT_DIR, 'mood_mturk.csv'))
        
    else:
        mood_pred = pd.read_csv(join(ROOT, 'results', 'mood_results.csv'),
                                index_col='creative')
    
    # results
    print("Summarizing results... ", end='', flush=True)
    
    # MTurk data
    mood_mturk = pd.read_csv(join(MT_DIR, 'mood_mturk.csv'), index_col='creative')
    
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
    
    # statistics
    with open(join(ROOT, 'statistics', 'mood_results.txt'), 'w') as fh:
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