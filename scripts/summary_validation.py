import cv2
import numpy as np
import os
import pandas as pd


from campvideo.video import Video
from functools import partial
from os.path import abspath, join, dirname
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# feature data directory
FEAT_DIR = join(ROOT, 'data', 'features')

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

# script for summarizing a collection of videos and computing statistics
def main():
    # create list of video paths for sample
    fnames = os.listdir(TRUE_DIR)
    fpaths = [join(ROOT, 'data', 'videos', fname + '.mp4') for fname in fnames]
    n = len(fpaths)
        
    print("Reading in video summaries...", end='', flush=True)
    # read in data
    true = pd.read_csv(join(TRUE_DIR, 'keyframes.csv'), index_col=['uid'], 
                       usecols=['uid', 'keyframes'])
    
    auto = pd.read_csv(join(FEAT_DIR, 'features.csv'), index_col=['uid'],
                       usecols=['uid', 'keyframes']).loc[true.index]
    
    # get keyframes
    true_inds = [[int(kf) for kf in keyframes.split(',')] 
                 for keyframes in true.keyframes]
    auto_inds = [[int(kf) for kf in keyframes.split(',')] 
                 for keyframes in auto.keyframes]
    print("Done!")
        
    # compute objective function values for summaries if specified
    print("Computing summary statistics...")
    
    # compute objective function component values
    r_auto, r_true, r_full = np.zeros(n), np.zeros(n), np.zeros(n)
    u_auto, u_true, u_full = np.zeros(n), np.zeros(n), np.zeros(n)
    n_auto, n_true, n_full = np.arange(n), np.arange(n), np.arange(n)
    for i, (fname, fpath, kf_auto, kf_true) in enumerate(zip(fnames, fpaths, auto_inds, true_inds)):
        end = '\r' if i < n-1 else '\n'
        print("\tProcessing video %d of %d..." % (i+1, n), end=end, flush=True)
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
    res.to_csv(join(ROOT, 'results', 'summary_results.csv'), index_label='uid')
    print("Done!")

if __name__ == '__main__':
    main()