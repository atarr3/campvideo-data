import json
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from campvideo.audio import Audio
from campvideo.image import Keyframes
from campvideo.video import Video
from os.path import abspath, join, dirname, exists
from string import punctuation

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# video directory
VID_DIR = join(ROOT, 'data', 'videos')

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# metadata
META_PATH = join(ROOT, 'data', 'metadata.csv')

# seed for replication, only affects summaries
SEED = 2002

# lookup table mapping YouTube IDs to CMAG IDs
with open(join(ROOT, 'data', 'matches', 'matches.json'), 'r') as fh:
    MATCHES = json.load(fh)
        
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-ow', '--overwrite', dest='overwrite', 
                        action='store_true', default=False, help='Flag for '
                        'specifying whether or not to overwite existing data. '
                        'If false, the script will only generate data when it '
                        'does not exist for the corresponding video. Specifying '
                        ' this flag will overwrite existing data')
    parser.add_argument('-ng', '--no-gcp', dest='gcp', 
                        action='store_false', default=True, help='Flag for '
                        'specifying whether to create results for tasks which '
                        'require GCP (image text detection and speech  '
                        'transcription). Specifying this flag will disable '
                        'these tasks')
    
    return parser.parse_args()

def main():
    # get command line arguments
    args = parse_arguments()  
    overwrite, gcp = args.overwrite, args.gcp
    
    # RNG
    rng = np.random.default_rng(SEED)
    
    # read in metadata, sorted alphabetically in ascending order
    meta = pd.read_csv(META_PATH, index_col='uid'
                       ).sort_index(key=lambda col: col.str.lower())    
    
    # process videos
    n = len(meta.index)
    for i, uid in enumerate(meta.index): 
        end = '\r' if i < n-1 else '\n'
        print('Processing video %d of %d...' %(i+1, n), end=end, flush=True)
        # get metadata and create Video object
        sub = meta.loc[uid]
        fpath = join(VID_DIR, uid + '.mp4') # video MUST have YT ID as title
        vid = Video(fpath)
        
        # output folder
        outfold = join(ROOT, 'data', 'intermediate', uid)
        if not exists(outfold): os.mkdir(outfold)
        
        # summary
        if not exists(join(outfold, 'keyframes.txt')) or overwrite:
            try:
                kf_ind = vid.summarize(rng=rng)
                # save
                with open(join(outfold, 'keyframes.txt'), 'w') as fh:
                    fh.write(','.join([str(e) for e in kf_ind]))
            except Exception:
                continue # must skip, need kf for later tasks
        else:
            # read in existing summary
            with open(join(outfold, 'keyframes.txt'), 'r') as fh:
                kf_ind = [int(e) for e in fh.read().split(',')]
        
        # audio feature
        if not exists(join(outfold, 'audiofeat.npy')) or overwrite:
            try:
                feat = Audio(fpath, fs=22050, nfft=1024, pre_emph=0.95
                            ).audiofeat(feature_set='best')
                # save
                with open(join(outfold, 'audiofeat.npy'), 'wb') as fh:
                    np.save(fh, feat)
            except Exception:
                pass
        
        # transcript
        if gcp:
            if not exists(join(outfold, 'transcript.txt')) or overwrite:
                try:
                    phrases = sub.fav_name.split(',') + sub.opp_names.split(',')
                    transcript = vid.transcribe(phrases=phrases)
                    # save
                    with open(join(outfold, 'transcript.txt'), 'w') as fh:
                        fh.write(transcript)
                except Exception:
                    pass
            
            # image text
            if not exists(join(outfold, 'imtext.txt')) or overwrite:
                try:
                    kf = Keyframes.fromvid(fpath, kf_ind=kf_ind)
                    imtext = kf.image_text()
                    # save
                    with open(join(outfold, 'imtext.txt'), 'wb') as fh:
                        # fancy string
                        combined = '\n'.join(
                            [' '.join(
                                [item.strip(punctuation) for item in items]
                                     ).lower() 
                             for items in imtext])
                        fh.write(combined.encode('utf-8'))
                except Exception:
                    pass

if __name__ == '__main__':
    main()      