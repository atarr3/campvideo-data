import json
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from campvideo.audio import Audio
from campvideo.image import Keyframes
from campvideo.video import Video
from os.path import join
from string import punctuation

# root folder for replication repo
ROOT = '..'

# video directory
VID_DIR = join(ROOT, 'data', 'youtube')

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# auxiliary data directory
AUX_DIR = join(ROOT, 'data', 'auxiliary')

# metadata
META_PATH = join(AUX_DIR, 'metadata.csv')

# seed for replication, only affects summaries
SEED = 2002

# lookup table mapping YouTube IDs to CMAG IDs
with open(join(AUX_DIR, 'matches.json'), 'r') as fh:
    MATCHES = json.load(fh)
        
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-ow', '--overwrite', dest='overwrite', 
                        action='store_true', default=False, help='flag for '
                        'specifying whether or not to overwite existing data. '
                        'If false, the script will only generate data when it '
                        'does not exist for the corresponding video.')
    parser.add_argument('-ng', '--no-gcp', dest='gcp', 
                        action='store_false', default=True, help='flag for '
                        'specifying whether to create results for tasks which '
                        'require GCP (image text detection and speech  '
                        'transcription). Specifying this flag will disable '
                        'these tasks')
    
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_arguments()  
    overwrite, gcp = args.overwrite, args.gcp
    
    # audio feature dimension
    d = 452
    
    # read in metadata, sort by `creative`
    meta = pd.read_csv(META_PATH, index_col=['creative', 'uid']).sort_index() 
    n = len(meta.index)
    
    # RNG
    rng = np.random.default_rng(SEED)
    # random array of seeds, one for each video
    seeds = rng.integers(0, high=2**32, size=n)
    
    # audio feature column names
    mfeat_names = ['v{}'.format(num) for num in range(d)]
    
    # # full column names
    # col_names = ['transcript', 'imtext', 'keyframes'] + mfeat_names
    
    # # datatype map
    # dtypes = dict(zip(col_names, [object]*3 + [np.float64] * d))
    
    # # create new dataframe if none exists
    # if not exists(join(AUX_DIR, 'features.csv')):
    #     # initialize dataframe
    #     feats = pd.DataFrame({'transcript': pd.NA, 'imtext': pd.NA, 'keyframes': pd.NA}, 
    #                           index=meta.index, dtype=str)
    #     # add audio feature columns
    #     feats = feats.reindex(columns=feats.columns.to_list() + mfeat_names)
    
    # read in existing data (imtext and transcript), add columns for keyframes and audio features
    feats = pd.read_csv(join(AUX_DIR, 'features_gcp.csv'), index_col=['creative', 'uid'])
    feats['keyframes'] = ''
    feats = feats.join(pd.DataFrame(data=np.nan, index=meta.index, columns=mfeat_names))
    
    # process videos
    failed = []
    for i, (creative, uid) in enumerate(meta.index): 
        end = '\r' if i < n-1 else '\n'
        print('Processing video %d of %d...' %(i+1, n), end=end, flush=True)
        # get metadata and create Video object
        sub = meta.loc[(creative, uid)]
        fpath = join(VID_DIR, uid + '.mp4') # video MUST have YT ID as title
        vid = Video(fpath)
        
        # summary
        if pd.isna(feats.loc[(creative, uid), 'keyframes']) or overwrite:
            try:
                # compute keyframes
                kf_ind = vid.summarize(rng=np.random.default_rng(seeds[i]))
                # store results
                feats.loc[(creative, uid), 'keyframes'] = ','.join([str(e) for e in kf_ind])
            except Exception as e: 
                # add video / step / message to failed array
                failed.append((uid, 'summarization', str(e)))
                
                # go to next video
                continue
        # read in keyframes for other tasks
        else:
            kf_ind = feats.loc[(creative, uid)].keyframes.split(',')
        
        
        # audio feature
        if pd.isna(feats.loc[(creative, uid), 'v0']) or overwrite: # checks first dimension only
            try:
                feat = Audio(fpath, fs=22050, nfft=1024, pre_emph=0.95
                            ).audiofeat(feature_set='best')
                # store results
                feats.loc[(creative, uid), mfeat_names] = feat
            except Exception as e:
                # add video / step / message to failed array
                failed.append((uid, 'audio feature extraction', str(e)))
        
        if gcp:
            # transcript
            if pd.isna(feats.loc[(creative, uid), 'transcript']) or overwrite:
                try:
                    phrases = sub.fav_name.split(',') + sub.opp_names.split(',')
                    transcript = vid.transcribe(phrases=phrases)
                    # store results
                    feats.loc[(creative, uid), 'transcript'] = transcript if len(transcript) > 0 else '<NO_TRANSCRIPT>'
                except Exception as e:
                    # add video / step / message to failed array
                    failed.append((uid, 'speech transcription', str(e)))
            
            # image text
            if pd.isna(feats.loc[(creative, uid), 'imtext']) or overwrite:
                try:
                    kf = Keyframes.fromvid(fpath, kf_ind=kf_ind)
                    imtext = kf.image_text()
                    # store results
                    combined = ' '.join([
                        ' '.join([item.strip(punctuation) for item in items])
                        for items in imtext if len(items) > 0]).lower()
                    feats.loc[(creative, uid), 'imtext'] = combined if len(combined) > 0 else '<NO_IMAGE_TEXT>'
                except Exception as e:
                    # add video / step / message to failed array
                    failed.append((uid, 'image text recognition', str(e)))
                    
    # save results with utf-8 encoding
    feats.to_csv(join(AUX_DIR, 'features_full.csv'), encoding='utf-8')
        
    # error log
    message = '\n'.join(
            ["- Video `{}.mp4` failed during {} step with error: `{}`".format(*error) 
             for error in failed]
            )
    if len(message) > 0:
        with open(join(ROOT, 'results', 'error_log.txt'), 'w') as fh:
            print(message, file=fh)

if __name__ == '__main__':
    main()      