import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from matplotlib import rc
from os.path import join
from sklearn.metrics import auc, roc_curve

# root folder for replication repo
ROOT = '..'

# seed for replication
SEED = 2002

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# validation
VAL_DIR = join(ROOT, 'data', 'validation')

# function for reading in WMP / CMAG data
def read_wmp():
    # open file
    wmp = pd.read_csv(join(WMP_DIR, 'wmp_final.csv'), index_col='creative')
    
    # sort index and return
    return wmp.sort_index()

def main():
    # read in WMP data and filter down to senate elections
    wmp = read_wmp()
    senate = wmp.loc[wmp.category == 'US SENATE', ['vid', 'f_picture', 'o_picture']]
    
    # remove videos missing labels for both vid and f_picture or o_picture
    senate = senate.loc[(~senate.vid.isna() | ~senate.f_picture.isna()) &
                         ~senate.o_picture.isna()]
    
    # wmp values
    o_wmp = senate.o_picture
    f_wmp = np.logical_or(senate.vid.fillna(0), senate.f_picture).astype(float)
    
    # read in our data
    facerec = pd.read_csv(join(ROOT, 'data', 'facerec_results.csv'), 
                          index_col='creative')
    
    # minimum encoding distances
    fmins, omins = facerec.f_dist.to_numpy(), facerec.o_dist.to_numpy()
    
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
    
    # ROC curves
    plt.rcParams['font.sans-serif'] = 'Verdana'
    #rc('text', usetex=True)

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
    plt.savefig(join(ROOT, 'results', 'figs','figureS13-8.pdf'), dpi=200, bbox_inches='tight') 
    
if __name__ == '__main__':
    main()