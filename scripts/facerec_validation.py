import json
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from campvideo.image import Keyframes
from os.path import abspath, join, dirname
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# video directory
VID_DIR = join(ROOT, 'data', 'videos')

# feature data directory
FEAT_DIR = join(ROOT, 'data', 'features')

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# ID encodings
ID_DIR = join(ROOT, 'data', 'ids')

# validation
VAL_DIR = join(ROOT, 'data', 'validation')

# metadata
META_PATH = join(ROOT, 'data', 'metadata.csv')

# seed
SEED = 2002

# lookup table mapping YouTube IDs to CMAG IDs
with open(join(ROOT, 'data', 'matches', 'matches.json'), 'r') as fh:
    MATCHES = json.load(fh)

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
                        'specifying to not detect and compute face encodings and '
                        'distances. If specified, the script will '
                        'check the `results` folder for a file called '
                        '`facerec_results.csv`.')
    
    return parser.parse_args()

def main():
    # read in CL arguments
    args = parse_arguments()
    calculate = args.calculate
    
    # read in WMP data and filter down to senate elections
    wmp = read_wmp()
    senate = wmp.loc[wmp.category == 'US SENATE', ['vid', 'f_picture', 'o_picture']]
    
    # remove videos missing labels for both vid and f_picture or o_picture
    senate = senate.loc[(~senate.vid.isna() | ~senate.f_picture.isna()) &
                         ~senate.o_picture.isna()]
    
    # metadata
    meta = pd.read_csv(META_PATH, index_col='creative')
    
    # detect opponent and favored appearances
    if calculate:
        # read in feature data
        feat = pd.read_csv(join(FEAT_DIR, 'features.csv'), index_col=['creative'],
                           usecols=['creative', 'keyframes'])      
        n = len(senate.index)
        fmins = np.ones(n)
        omins = np.ones(n)
        for i, creative in enumerate(senate.index):
            end = '\r' if i < n-1 else '\n'
            print('Processing video %d of %d...' %(i+1, n), end=end, flush=True)
            
            # metadata 
            metadata = meta.loc[creative]
            uid = metadata.uid
            fav_path = metadata.fav_path
            opp_paths = metadata.opp_paths
            
            # construct keyframes object
            vpath = join(VID_DIR, uid + '.mp4')
            kf_ind = [int(e) for e in feat.loc[creative].keyframes.split(',')]
            kf = Keyframes.fromvid(vpath, kf_ind=kf_ind, max_dim=1280)
            
            # load in encodings
            if not pd.isnull(fav_path):
                fav_path = join(ROOT, fav_path)
                with open(fav_path, 'rb') as fh:
                    fav_enc = np.load(fh)
            else:
                fav_enc = None
                
            if not pd.isnull(opp_paths):
                opp_encs = []
                for opp_path in opp_paths.split(','):
                    opp_path = join(ROOT, opp_path)
                    with open(opp_path, 'rb') as fh:
                        opp_encs.append(np.load(fh))
            else:
                opp_encs = None
                
            # get labels and distances
            if fav_enc is not None:
                fdist, _ = kf.facerec(fav_enc, return_dists=True)
                fmins[i] = fdist.min() if len(fdist) > 0 else 1.
                
            if opp_encs is not None:
                for opp_enc in opp_encs:
                    odist, _ = kf.facerec(opp_enc, return_dists=True)
                    omins[i] = min(omins[i], odist.min()) if len(odist) > 0 else omins[i]
        
        # compute threshold
        xo_train, xo_test, yo_train, yo_test = train_test_split(omins.reshape(-1,1),
                                                                senate.o_picture,
                                                                test_size=0.2,
                                                                random_state=SEED)
        
        params = [{'C': np.geomspace(0.01, 10, 25)}]
        grid_o = GridSearchCV(LinearSVC(dual=False),
                              params, scoring='accuracy', cv=5)
        grid_o.fit(xo_train, yo_train)
        svm_o = grid_o.best_estimator_
        
        # distance threshold
        thr_o = -svm_o.intercept_[0] / svm_o.coef_[0][0]
        
        # predictions
        o_pred = (omins < thr_o).astype(int)
        f_pred = (fmins < thr_o).astype(int)
                    
        # save results
        facerec = pd.DataFrame({'uid': meta.loc[senate.index].uid, 
                                'f_picture': f_pred, 'f_dist': fmins, 
                                'o_picture': o_pred, 'o_dist': omins}, 
                               index=senate.index)
        facerec.to_csv(join(ROOT, 'results', 'facerec_results.csv'), index=True)
        print("Done!")
    else:
        facerec = pd.read_csv(join(ROOT, 'results', 'facerec_results.csv'), 
                              index_col='creative')
        
        # minimum encoding distances
        fmins, omins = facerec.f_dist.to_numpy(), facerec.o_dist.to_numpy()
            
        # compute threshold
        xo_train, xo_test, yo_train, yo_test = train_test_split(omins.reshape(-1,1),
                                                                senate.o_picture,
                                                                test_size=0.2,
                                                                random_state=SEED)
        
        params = [{'C': np.geomspace(0.01, 10, 25)}]
        grid_o = GridSearchCV(LinearSVC(dual=False),
                              params, scoring='accuracy', cv=5)
        grid_o.fit(xo_train, yo_train)
        svm_o = grid_o.best_estimator_
        
        # distance threshold
        thr_o = -svm_o.intercept_[0] / svm_o.coef_[0][0]
        
        # predicted values
        o_pred = (omins < thr_o).astype(int)
        f_pred = (fmins < thr_o).astype(int)
    
    print("Summarizing results... ", end='', flush=True)
    
    # wmp values
    o_wmp = senate.o_picture
    f_wmp = np.logical_or(senate.vid.fillna(0), senate.f_picture).astype(float)
    
    # corrected wmp values
    o_corr = pd.read_csv(join(VAL_DIR, 'o_picture_validation.csv'), 
                         index_col='creative')
    f_corr = pd.read_csv(join(VAL_DIR, 'f_picture_validation.csv'), 
                         index_col='creative')
    
    ocorr_ind = o_corr.index[o_corr.note == "wmp wrong"]
    fcorr_ind = f_corr.index[f_corr.note.isin(["wmp wrong", "definition"])]
    
    o_wmp_corr = o_wmp.copy()
    o_wmp_corr[ocorr_ind] = (o_wmp_corr[ocorr_ind] == 0).astype(int)
    f_wmp_corr = f_wmp.copy()
    f_wmp_corr[fcorr_ind] = (f_wmp_corr[fcorr_ind] == 0).astype(int)
    
    # new threshold
    xo_train, xo_test, yo_train, yo_test = train_test_split(omins.reshape(-1,1),
                                                            o_wmp_corr,
                                                            test_size=0.2,
                                                            random_state=SEED)
    grid_o.fit(xo_train, yo_train)
    svm_o = grid_o.best_estimator_
    
    # distance threshold
    thr_o_corr = -svm_o.intercept_[0] / svm_o.coef_[0][0]
    
    # predicted values
    o_pred_corr = (omins < thr_o_corr).astype(int)
    f_pred_corr = (fmins < thr_o_corr).astype(int)
    
    # confusion matrices
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    # corrected
    cm_o_corr = pd.DataFrame(
                    confusion_matrix(o_wmp_corr, o_pred_corr, normalize='all'),
                    columns=cols, index=rows
                    )
    cm_f_corr = pd.DataFrame(
                    confusion_matrix(f_wmp_corr, f_pred_corr, normalize='all'),
                    columns=cols, index=rows
                    )
    
    ## statistics ##
    
    # number of disagreements
    n_opp = len(o_corr)
    n_fav = len(f_corr)
    
    # number of mistakes by wmp
    n_opp_mis = sum(o_corr.note == "wmp wrong")
    n_fav_mis = sum(f_corr.note == "wmp wrong")
    
    # number of mistakes due to definition
    n_fav_def = sum(f_corr.note == "definition")
    
    # precision, recall, accuracy (corrected)
    p_opp = precision_score(o_wmp_corr, o_pred_corr)
    p_fav = precision_score(f_wmp_corr, f_pred_corr)
    r_opp = recall_score(o_wmp_corr, o_pred_corr)
    r_fav = recall_score(f_wmp_corr, f_pred_corr)
    a_opp = accuracy_score(o_wmp_corr, o_pred_corr)
    a_fav = accuracy_score(f_wmp_corr, f_pred_corr)
        
    # facerec statistics
    with open(join(ROOT, 'performance', 'facerec_results.txt'), 'w') as fh:
        print("Thresholds", file=fh)
        print("----------", file=fh)
        print(" Original: {:.4}".format(thr_o), file=fh)
        print("Corrected: {:.4}".format(thr_o_corr), file=fh)
        print(file=fh)
        print("Mistake Analysis", file=fh)
        print("----------------", file=fh)
        print("f_picture disagreements: %d" % n_fav, file=fh)
        print("o_picture disagreements: %d" % n_opp, file=fh)
        print(file=fh)
        print("f_picture mistakes (WMP): {} ({:.0%})".format(n_fav_mis, 
                                                             n_fav_mis / n_fav), 
              file=fh)
        print("f_picture mistakes (def): {} ({:.0%})".format(n_fav_def, 
                                                             n_fav_def / n_fav), 
              file=fh)
        n_fav_oth = n_fav - (n_fav_def + n_fav_mis)
        print("f_picture mistakes (oth): {} ({:.0%})".format(n_fav_oth, 
                                                             n_fav_oth / n_fav), 
              file=fh)
        print(file=fh)
        print("o_picture mistakes (WMP): {} ({:.0%})".format(n_opp_mis, 
                                                             n_opp_mis / n_opp), 
              file=fh)
        n_opp_oth = n_opp - n_opp_mis
        print("o_picture mistakes (oth): {} ({:.0%})".format(n_opp_oth, 
                                                             n_opp_oth / n_opp), 
              file=fh)
        print(file=fh)
        print("Favored Candidate (Corrected)", file=fh)
        print("-----------------------------", file=fh)
        print(cm_f_corr, file=fh)
        print(file=fh)
        print("Opposing Candidate (Corrected)", file=fh)
        print("------------------------------", file=fh)
        print(cm_o_corr, file=fh)
        print(file=fh)
        print("Classification Metrics", file=fh)
        print("----------------------", file=fh)
        print(" Favored Pre, Rec, Acc: {:.2}, {:.2}, {:.2}".format(p_fav, r_fav, 
                                                                   a_fav), 
              file=fh)
        print("Opposing Pre, Rec, Acc: {:.2}, {:.2}, {:.2}".format(p_opp, r_opp, 
                                                                   a_opp),
              file=fh)           

    print("Done!")
if __name__ == '__main__':
    main()