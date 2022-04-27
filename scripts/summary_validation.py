import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from campvideo.video import Video
from functools import partial
from matplotlib import rc
from os.path import abspath, join, dirname
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# output directory for saving/loading keyframe results
AUTO_DIR = join(ROOT, 'data', 'intermediate')

# manually created summaries
TRUE_DIR = join(ROOT, 'data', 'validation', 'summary')
   
# computes representativeness and uniqueness
def objective(kf_ind, w, d):
# representativeness
    r = w[kf_ind].max(axis=0).sum()
    # uniqueness
    d_sub = d[np.ix_(kf_ind, kf_ind)]
    u = 0
    for j in range(1, len(d_sub)):
        # minumum distance to all previously added frames
        u += d_sub[j, :j].min()
        
    return r,u

def parse_arguments():
    parser = argparse.ArgumentParser()  
    # parser.add_argument('-ns', '--no-summarize', dest='summarize', 
    #                     action='store_false', default=True, help='Flag for '
    #                     'specifying whether or not compute video summaries. '
    #                     'If specified, the script will look for summaries in '
    #                     'the `automated` folder in the replication directory.')
    parser.add_argument('-nc', '--no-calculate', dest='calculate', 
                        action='store_false', default=True, help='Flag for '
                        'specifying whether or not compute summary statistics. '
                        'If specified, the script will look for the results in'
                        ' `results/summary_validation.csv`.')

    return parser.parse_args()

# script for summarizing a collection of videos and computing statistics
def main():
    args = parse_arguments()
    calculate = args.calculate
    
    # create list of video paths for sample
    with open(join(TRUE_DIR, 'sample.txt'), 'r') as fh:
        fnames = fh.read().splitlines()
        fpaths = [join(ROOT, 'data', 'videos', fname + '.mp4') 
                                                          for fname in fnames]
    
    # # compute video summaries if specified
    # n = len(fpaths)
    # if summarize:
    #     print("Summarizing videos...")
    #     auto_inds = []
    #     for i, (fname, fpath) in enumerate(zip(fnames, fpaths)):
    #         end = '\r' if i < n-1 else '\n'
    #         print("    Processing video %d of %d..." % (i+1, n), end=end, flush=True)
    #         vid = Video(fpath)
    #         kf_ind = vid.summarize(rng=rng)
    #         auto_inds.append(kf_ind)
    #         # save
    #         with open(join(AUTO_DIR, fname, 'keyframes.txt'), 'w') as fh:
    #             fh.write(','.join([str(e) for e in kf_ind]))
    #     print("Done!\n")
    # # otherwise read in summaries
    # else:
        
    print("Reading in video summaries...")
    auto_inds = []
    n = len(fpaths)
    for (fname, fpath) in zip(fnames, fpaths):
        with open(join(AUTO_DIR, fname, 'keyframes.txt'), 'r') as fh:
            auto_inds.append([int(e) for e in fh.read().split(',')])
    print("Done!\n")
        
    # compute objective function values for summaries if specified
    if calculate:
        print("Computing summary statistics...")
        # read in keyframe indices for manually created summaries
        true_inds = []
        for fname in fnames:
            with open(join(TRUE_DIR, fname, 'keyframes.txt'), 'r') as fh:
                true_inds.append([int(e) for e in fh.read().split(',')])
        
        # compute objective function component values
        r_auto, r_true, r_full = np.zeros(n), np.zeros(n), np.zeros(n)
        u_auto, u_true, u_full = np.zeros(n), np.zeros(n), np.zeros(n)
        n_auto, n_true, n_full = np.arange(n), np.arange(n), np.arange(n)
        for i, (fname, fpath, kf_auto, kf_true) in enumerate(zip(fnames, fpaths, auto_inds, true_inds)):
            end = '\r' if i < n-1 else '\n'
            print("    Processing video %d of %d..." % (i+1, n), end=end, flush=True)
            # get video features
            vid = Video(fpath)
            labhist, hog = vid.videofeats()
            
            # subset down to non-monochromatic frames
            index_sub = np.where(~np.isnan(labhist).any(axis=1))[0]
            labhist_nm = labhist[index_sub]
            hog_nm = hog[index_sub]
            
            # re-index to index_sub position
            kf_auto = np.sort([np.where(index_sub == ele)[0][0] for ele in kf_auto])
            kf_true = np.sort([np.where(index_sub == ele)[0][0] for ele in kf_true])
            kf_full = np.arange(len(index_sub))
            
            # pairwise distances
            cfunc = partial(cv2.compareHist, method=cv2.HISTCMP_CHISQR_ALT)       
            w = cosine_similarity(hog_nm)
            d = 0.25 * pairwise_distances(labhist_nm, metric=cfunc, n_jobs=-1)
            
            # number of frames in summary
            n_auto[i] = kf_auto.size
            n_true[i] = kf_true.size
            n_full[i] = kf_full.size
            
            # uniqueness and representativeness
            r_auto[i], u_auto[i] = objective(kf_auto, w, d)
            r_true[i], u_true[i] = objective(kf_true, w, d)
            r_full[i], u_full[i] = np.float(len(w)), objective(kf_full, w, d)[1]
        
        # build dataframe
        res = pd.DataFrame(data={'r_auto': r_auto, 'r_true': r_true, 'r_full': r_full,
                                 'u_auto': u_auto, 'u_true': u_true, 'u_full': u_full,
                                 'n_auto': n_auto, 'n_true': n_true, 'n_full': n_full},
                           index=fnames)
        
        # save dataframe
        res.to_csv(join(ROOT, 'results', 'summary_validation.csv'), index_label='uid')
        print("Done!\n")
    # read in data
    else:
        print("Reading in summary statistics...")
        res = pd.read_csv(join(ROOT, 'results', 'summary_validation.csv'),
                          index_col='uid')
        print("Done!\n")
    # plot results
    print("Plotting results...")
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    constrained_layout = True # use tight layout if False
    
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 4.333), sharey=False, 
                            constrained_layout=constrained_layout)
    
    for i in range(3):
        # plot scatter plot of n_true vs n_auto
        if i == 0:
            ticks = [0, 10, 20]
            # scatter plot and y=x line
            axs[i].scatter(res.n_auto, res.n_true, s=10, c='black')
            axs[i].plot([0, 25], [0, 25], 'k--', lw=1)
            
            # axis formatting
            axs[i].set_xlim(0, 25)
            axs[i].set_ylim(0, 25)
            axs[i].set_xlabel('Automated Summary', fontsize=12)
            axs[i].set_ylabel('Manual Summary', fontsize=12)
            axs[i].tick_params(labelsize=10)
            axs[i].set_xticks(ticks)
            axs[i].set_yticks(ticks)
            axs[i].set_aspect(1 / axs[i].get_data_ratio())
        
        # smoothed histogram of represenativeness, normalized by number of frames 
        elif i == 1:
            # histogram plot
            sns.distplot(res.r_auto / res.n_full, hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 1}, ax=axs[i]
                        )
            sns.distplot(res.r_true / res.n_full, hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 1}, ax=axs[i]
                        )
            
            # text labels
            axs[i].text(0.9, 10, 'Automated', fontsize=8)
            axs[i].text(0.74, 5.8, 'Manual', fontsize=8)
            
            # axis formatting
            axs[i].set_xlabel('Representativeness', fontsize=12)
            axs[i].set_ylabel('Density', fontsize=12)
            axs[i].tick_params(labelsize=10)
            axs[i].set_xticks([0.7, 0.8, 0.9, 1.0])
            axs[i].set_yticks([0, 3, 6, 9, 12])
            axs[i].set_aspect(1 / axs[i].get_data_ratio())
            
        # smoothed histogram of uniqueness
        else:
            # histogram plot
            sns.distplot(res.u_auto / res.n_auto, hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 1}, ax=axs[i]
                        )
            sns.distplot(res.u_true / res.n_true, hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 1}, ax=axs[i]
                        )
            
            # text labels
            axs[i].text(0.02, 2.5, 'Automated', fontsize=8)
            axs[i].text(0.5, 1.8, 'Manual', fontsize=8)
            
            # axis formatting
            axs[i].set_xlabel('Uniqueness', fontsize=12)
            axs[i].set_ylabel('Density', fontsize=12)
            axs[i].set_xlim(0, 0.74)
            axs[i].tick_params(labelsize=10)
            axs[i].set_xticks([0.0, 0.2, 0.4, 0.6])
            axs[i].set_yticks([0, 1, 2, 3])
            axs[i].set_aspect(1 / axs[i].get_data_ratio())
    
    # layout and save
    if not constrained_layout: fig.tight_layout()
    plt.savefig(join(ROOT, 'figs', 'figS7-4.pdf'), dpi=200, bbox_inches='tight')
    print("Done!")

if __name__ == '__main__':
    main()