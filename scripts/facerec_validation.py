import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from campvideo.image import Keyframes
from matplotlib import rc
from os.path import abspath, join, dirname
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score 
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# video directory
VID_DIR = join(ROOT, 'data', 'videos')

# intermediate data
INT_DIR = join(ROOT, 'data', 'intermediate')

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
                        'specifying whether or not to detect and compute face '
                        'encodings and distances. If False, the script will '
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
            spath = join(INT_DIR, uid, 'keyframes.txt')
            
            with open(spath, 'r') as fh:
                kf_ind = [int(e) for e in fh.read().split(',')]
                
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
        facerec = pd.DataFrame({'f_picture': f_pred, 'f_dist': fmins, 
                                'o_picture': o_pred, 'o_dist': omins}, 
                               index=senate.index)
        facerec.to_csv(join(ROOT, 'results', 'facerec_results.csv'), index=True)
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
    
    # original
    cm_o = pd.DataFrame(
                confusion_matrix(o_wmp, o_pred, normalize='all'),
                columns=cols, index=rows
                )
    cm_f = pd.DataFrame(
                confusion_matrix(f_wmp, f_pred, normalize='all'),
                columns=cols, index=rows
                )
    
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
    
    # confusion matrix (uncorrected, Table 4)
    with open(join(ROOT, 'tables', 'table4.txt'), 'w') as fh:
        print("Favored Candidate", file=fh)
        print("-----------------", file=fh)
        print(cm_f, file=fh)
        print(file=fh)
        print("Opposing Candidate", file=fh)
        print("------------------", file=fh)
        print(cm_o, file=fh)
        
    # facerec statistics
    with open(join(ROOT, 'results', 'facerec_results.txt'), 'w') as fh:
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
    
    # ROC curves
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')

    # original
    fpr_orig_f, tpr_orig_f, _ = roc_curve(f_wmp, fmins, pos_label=0)
    auc_orig_f = auc(fpr_orig_f, tpr_orig_f)
    fpr_orig_o, tpr_orig_o, _ = roc_curve(o_wmp, omins, pos_label=0)
    auc_orig_o = auc(fpr_orig_o, tpr_orig_o)

    # corrected
    fpr_corr_f, tpr_corr_f, _ = roc_curve(f_wmp_corr, fmins, pos_label=0)
    auc_corr_f = auc(fpr_corr_f, tpr_corr_f)
    fpr_corr_o, tpr_corr_o, _ = roc_curve(o_wmp_corr, omins, pos_label=0)
    auc_corr_o = auc(fpr_corr_o, tpr_corr_o)
    
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 4.333), sharey=True, 
                            constrained_layout=True)

    # ticks
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # original data
    axs[0].plot(fpr_orig_f, tpr_orig_f, label=r'Favored (AUC = %0.2f)' % auc_orig_f, 
                lw=1.5, alpha=1)
    axs[0].plot(fpr_orig_o, tpr_orig_o, label=r'Opponent (AUC = %0.2f)' % auc_orig_o, 
                lw=1.5, alpha=1)
    axs[0].set_xlabel(r'False Positive Rate', fontsize=15)
    axs[0].set_ylabel(r'True Positive Rate', fontsize=15)
    axs[0].set_xticks(ticks)
    axs[0].set_yticks(ticks)
    axs[0].tick_params(labelsize=12)
    axs[0].set_aspect(1 / axs[0].get_data_ratio())    
    axs[0].grid(True)
    axs[0].legend(fontsize=9, framealpha=1)

    # corrected data
    axs[1].plot(fpr_corr_f, tpr_corr_f, label=r'Favored (AUC = %0.2f)' % auc_corr_f, 
                lw=1.5, alpha=1)
    axs[1].plot(fpr_corr_o, tpr_corr_o, label=r'Opponent (AUC = %0.2f)' % auc_corr_o, 
                lw=1.5, alpha=1)
    axs[1].set_xlabel(r'False Positive Rate', fontsize=15)
    axs[1].set_xticks(ticks)
    axs[1].set_yticks(ticks)
    axs[1].tick_params(labelsize=12)
    axs[1].set_aspect(1 / axs[1].get_data_ratio())    
    axs[1].grid(True)
    axs[1].legend(fontsize=9, framealpha=1)

    fig.set_constrained_layout_pads(wspace=0.2)
    plt.savefig(join(ROOT, 'figs','figS13-8.pdf'), dpi=200, bbox_inches='tight')            

if __name__ == '__main__':
    main()