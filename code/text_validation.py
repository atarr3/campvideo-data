import cv2
import json
import numpy as np
import pandas as pd
import re
import spacy

from argparse import ArgumentParser
from campvideo import Keyframes, Text
from os.path import join
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from string import punctuation

# root folder for replication repo
ROOT = '..'

# video directory
VID_DIR = join(ROOT, 'data', 'youtube')

# auxiliary data directory
AUX_DIR = join(ROOT, 'data', 'auxiliary')

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

# validation
VAL_DIR = join(ROOT, 'data', 'validation')

# MTurk
MTURK_DIR = join(ROOT, 'data', 'mturk')

# ID encodings
ID_DIR = join(AUX_DIR, 'ids')

# metadata
META_PATH = join(AUX_DIR, 'metadata.csv')

# issue vocabulary list
VOCAB_PATH = join(AUX_DIR, 'issuenames.csv')
VOCAB = pd.read_csv(VOCAB_PATH)

# seed
SEED = 2002

# lookup table mapping YouTube IDs to CMAG IDs
with open(join(AUX_DIR, 'matches.json'), 'r') as fh:
    MATCHES = json.load(fh)
    
# lookup table mapping CMAG IDs to YouTube IDs
MATCHES_CMAG = dict(zip(MATCHES.values(), MATCHES.keys()))

# Modified list of stop words taken from
# https://www.ranks.nl/stopwords
STOP = frozenset([
    "approve","message",
    "a", "about", "am", "an", "and", "any", "are", "as", "at", "be", 
    "been",  "being", "both", "by",  "during", "each", "for", "from", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "let's", "me",  "must", "my", "myself", "nor",  "of", "once", "or", "other", 
    "ought", "ourselves", "own", "shall", "she", "should", "so", "some", "such", 
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "to", "until", "up", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "would", 
    "you", "your", "yours", "yourself", "yourselves", "'s"])

# list of generic entity descriptors
ENTS = frozenset(["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL",
                  "CARDINAL"])

# list of names to keep when removing names
NAMES = frozenset(["barack", "obama", "barack obama", "pelosi", "nancy pelosi",
                   "reagan", "ronald reagan"])

# spacy NLP parser
NLP = spacy.load('en_core_web_md')

# regular expression for removing apostrophes, but preserving those in names
P = re.compile("([\w]{2})'.*$")

# function for checking if any name in NAMES is contained in input string
def has_name(text):
    return any([name in text for name in NAMES])

# tokenizer function
def tokenize(text, ner=True, keep_names=False, keep_pron=False):
    # parse text using spacy
    parsed = NLP(text)
    
    # split named entities
    if ner:
        # lemmatize non-entity tokens
        tokens = [token.lemma_ for token in parsed 
                                   if token.ent_type == 0]
        # convert to lowercase, excluding pronouns
        tokens = [token.lower() if token != "-PRON-" else token 
                            for token in tokens]
        
        # split names from rest of entities
        names = [ent for ent in parsed.ents if ent.label_ == "PERSON"]
        ents = [ent for ent in parsed.ents if ent.label_ != "PERSON"]
        
        # convert numeric entities to generic description and remove apostrophe
        # suffixes
        ents = [P.sub("\\1", ent.text.lower()) if ent.label_ not in ENTS 
                                         else '-' + ent.label_ + '-'
                                     for ent in ents]
        
        # remove apostrophe suffixes from names
        names = [P.sub("\\1", name.text.lower()) for name in names]
        
        # relabel names to -NAME- exluding major political figures
        if not keep_names:
            # names = [name if has_name(name) else '-NAME-' for name in names]
            names = [name for name in names if has_name(name)]   
    else: # ignore named entities
        # lemmatize all tokens
        tokens = [token.lemma_.lower() for token in parsed]
        
                  
    # keep pronouns and remove all other stop words / punctuation
    if keep_pron:
        tokens = [token for token in tokens
                            if token not in STOP and token not in punctuation]
    else:
        tokens = [token for token in tokens
                            if token not in STOP and token not in punctuation
                            and token != '-PRON-']
    
    # return list of tokens
    if ner:
        return tokens + ents + names
    else:
        return tokens
    
# dummy tokenizer
def dummy(tokens):
    return tokens
    
# class implementing naive Bayes for heterogenous data
class HeterogenousNB(BaseEstimator, ClassifierMixin):
    def __init__(self, discrete_clf='bernoulli', alpha=1.0, 
                 fit_prior=True, class_prior=None):
        
        self.discrete_clf = discrete_clf
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior

    def fit(self, X, y, split_ind=-1):
        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # check discrete_clf
        if self.discrete_clf.lower() not in ['bernoulli', 'multinomial']:
            raise ValueError("Incorrect classifier for discrete data, "
                             "please specify 'bernoulli' or 'multinomial'")
            
        # index marking split between discrete (1st segment) and continuous data
        self.split_ind_ = split_ind 
        
        # store the classes seen during fit
        self.classes_ = unique_labels(y)
    
        self.X_ = X
        self.y_ = y
        
        # fit naive Bayes classifiers
        X_disc, X_cont = np.split(X, np.atleast_1d(split_ind), axis=1)
        
        if self.discrete_clf == 'bernoulli':
            nb_disc = BernoulliNB(alpha=self.alpha, fit_prior=self.fit_prior,
                                  class_prior=self.class_prior).fit(X_disc, y)
        else:
            nb_disc = MultinomialNB(alpha=self.alpha, fit_prior=self.fit_prior,
                                    class_prior=self.class_prior).fit(X_disc, y) 
        nb_cont = GaussianNB(priors=self.class_prior).fit(X_cont, y)
        
        # save fitted classifiers as attributes
        self.nb_disc_ = nb_disc
        self.nb_cont_ = nb_cont
        
        # return the classifier
        return self

    def predict(self, X):
        # check is fit had been called
        check_is_fitted(self)
    
        # input validation
        X = check_array(X)
        
        # prior probs
        log_prior = self.nb_disc_.class_log_prior_
        
        # GNB params
        theta = self.nb_cont_.theta_
        sigma = self.nb_cont_.sigma_
        
        # BNB / MNB params
        logp = self.nb_disc_.feature_log_prob_
        
        # compute joint log-likelihood
        X_disc, X_cont = np.split(X, np.atleast_1d(self.split_ind_), axis=1)
        
        jll_disc = np.dot(X_disc, logp.T)
        jll_cont = []
        for i in range(np.size(self.classes_)):
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * sigma[i, :]))
            n_ij -= 0.5 * np.sum(((X_cont - theta[i, :]) ** 2) / 
                                 (sigma[i, :]), 1)
            jll_cont.append(n_ij)
            
        jll_cont = np.array(jll_cont).T
        
        # total joint log-likelihood
        jll_total = log_prior + jll_disc + jll_cont
    
        return self.classes_[np.argmax(jll_total, axis=1)]

# function for reading in WMP / CMAG data
def read_wmp():
    # open file
    wmp = pd.read_csv(join(WMP_DIR, 'wmp_final.csv'), index_col='creative')
    
    # sort index and return
    return wmp.sort_index()

# function for resizing image while preserving aspect-ratio
def resize_im(im, max_dim=1280):
    h, w,_ = im.shape
    # dont resize if image is below the maximum dimension
    if max(h, w) <= max_dim: 
        return im
    
    # compute aspect-ratio
    ar = w / h
    # scale to max dim
    new_w = max_dim if w >= h else int(max_dim * ar)
    new_h = max_dim if w < h else int(max_dim / ar)
    
    # return resized image
    return cv2.resize(im, (new_w,new_h))

# function for creating list of names to check in name mentions
def namegen(name, return_plurals=True):
    # name is a single string for the full name, no punctuation removed
    
    # return None if nan passed
    if type(name) is not str: return [None]
    
    # split names, ignore first name
    names = name.split(' ')[1:]
    
    # add hyphen to double last name
    name_hy = '-'.join(names) if len(names) == 2 else None
    
    # remove hyphen from names
    names_nohy = [name.split('-') if '-' in name else None for name in names]
    names_nohy = [' '.join(item) for item in names_nohy if item is not None]
    
    # final part of name for double last name
    if len(names) == 2:
        final = names[-1]
    elif '-' in names[0]:
        final = names[0].split('-')[-1]
    else:
        final = None
        
    # preliminary list
    gen_names = list(filter(None, [' '.join(names)] + [name_hy] + 
                                  names_nohy + [final]))
        
    # add pluralization (this seems like overkill on June's part)
    plurals = [item + 's' for item in gen_names if not item.endswith('s')]
    
    # NOTE: no need to add possessive, spacy tokenizes 's separately
    
    if return_plurals:
        return gen_names + plurals
    else:
        return gen_names

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-nc', '--no-calculate', dest='calculate', 
                        action='store_false', default=True, help='Flag for '
                        'specifying to not train text models and predict '
                        'labels. If specified, the script will '
                        'check the `data` folder for files called '
                        '`negativity_results.csv` and `mentions_results.csv`')
    parser.add_argument('-nm', '--no-mention', dest='im_flag', 
                        action='store_false', default=True, help='Flag for '
                        'specifying to not perform issue/opponent mentions')
    parser.add_argument('-nn', '--no-negativity', dest='an_flag', 
                        action='store_false', default=True, help='Flag for '
                        'specifying to not perform ad negativity '
                        'classification')
    
    return parser.parse_args()

def main():
    # read in CL arguments
    args = parse_arguments()
    calculate, im_flag, an_flag = args.calculate, args.im_flag, args.an_flag
    
    # metadata
    meta = pd.read_csv(META_PATH, index_col='creative')
    
    # obama face encoding
    with open(join(ID_DIR, 'obama.npy'), 'rb') as fh:
        obama_enc = np.load(fh)
    
    # read in WMP data
    wmp = read_wmp()
    
    # merge issue30 (abortion) and issue58 (women's health)
    wmp['issue30'] = np.logical_or(wmp.issue30, wmp.issue58).astype(int)
    # merge issue53 (healthcare) and issue59 (obamacare)
    # NOTE: issue59 only labeled in 2014 data
    wmp['issue53'] = np.logical_or(wmp.issue53, wmp.issue59, 
                                   out=wmp.issue53.to_numpy(),
                                   where=~wmp.issue59.isna())
    
    # subset to issue/o_mention columns
    iss_wmp = wmp.loc[:, VOCAB.wmp.to_list()].astype(int, errors='ignore')
    oment_wmp = wmp.loc[:, 'o_mention'].dropna().astype(int)
    
    # convert non-binary variables to binary (first 10 columns)
    iss_wmp.iloc[:, :10] = (~np.logical_or(iss_wmp.iloc[:, :10] == 0, 
                                           iss_wmp.iloc[:, :10] == 'No')).astype(int)
        
    # remove tonecmag observations with nan values
    tone = wmp.dropna(subset=['tonecmag'])
    # drop 'contrast' observations
    tone = tone.loc[tone.tonecmag != 'CONTRAST']
    # recast tonecmag to 1/0 for sentiment
    tone_wmp = ((tone.tonecmag == 'POS') | 
                (tone.tonecmag == 'POSITIVE')).astype(int)
        
    # process videos
    if calculate:
        # read in features (full)
        feat = pd.read_csv(join(AUX_DIR, 'features_full.csv'), index_col=['creative'])
        # replace na with empty strings
        feat = feat.fillna("")
        
        if im_flag:
            print("Detecting issue and opponent mentions...")
            n = len(iss_wmp.index)
            iss_pred = pd.DataFrame(0, dtype=int, columns=iss_wmp.columns,
                index=pd.MultiIndex.from_product([iss_wmp.index, ['text', 'both']],
                                                 names=['creative', 'feature'])
                )
            oment_pred = pd.Series(0, dtype=int, name='o_mention', 
                index=pd.MultiIndex.from_product([iss_wmp.index, ['text', 'both']],
                                                 names=['creative', 'feature']))
            
            for i, creative in enumerate(iss_wmp.index):
                end = '\r' if i < n-1 else '\n'
                print('\tProcessing video %d of %d...' %(i+1, n), end=end, flush=True)
                
                # metadata 
                metadata = meta.loc[creative]
                uid = metadata.uid
                opp_names = metadata.opp_names.split(',')
                
                # subset to current video data
                sub = feat.loc[creative]
                
                # read transcript and construct Text object
                trans_text = Text(sub.transcript.lower())
                
                # read image text and construct Text object
                im_text = Text(sub.imtext.replace('.', ' '))
                
                # read in keyframes and construct Keyframes object
                kf_ind = [int(e) for e in sub.keyframes.split(',')]
                    
                kf = Keyframes.fromvid(join(VID_DIR, uid + '.mp4'), kf_ind, max_dim=1280)
                
                # Obama face recognition
                obama_im = kf.facerec(obama_enc, dist_thr=0.5215)
                
                # issue mention
                iss_trans = trans_text.issue_mention(include_names=True, include_phrases=True)
                iss_im = im_text.issue_mention(include_names=True, include_phrases=True)
                
                # face recognition for obama variable (`prsment`)
                iss_im['prsment'] |= obama_im
                # ignore visual data for congress (`congmt`)  and wall street (`mention16`)
                iss_im['congmt'] = 0
                iss_im['mention16'] = 0
                
                # opponent mention
                opp_trans = int(any(trans_text.opp_mention(name) for name in opp_names)) 
                opp_im = int(any(im_text.opp_mention(name) for name in opp_names))
                
                # store in data frame
                iss_pred.loc[(creative, 'text')] = iss_trans
                oment_pred.loc[(creative, 'text')] = opp_trans
                iss_pred.loc[(creative, 'both')] = iss_trans | iss_im
                oment_pred.loc[(creative, 'both')] = opp_trans | opp_im
            
            # attach o_omention and uids to dataframe
            iss_pred['o_mention'] = oment_pred
            uids_iss = [MATCHES_CMAG[ele] for ele in iss_pred.index.get_level_values('creative')]
            
            iss_pred = iss_pred.reset_index()
            iss_pred.insert(1, 'uid', uids_iss) # after creative
            iss_pred.set_index(['creative', 'uid', 'feature'], inplace=True)
            iss_pred.to_csv(join(ROOT, 'data', 'mentions_results.csv'))
            
            # update MTurk results
            # iss_mturk = pd.read_csv(join(MTURK_DIR, 'issue_mturk.csv'), 
                                    # index_col=['creative', 'uid', 'issue'])
            # iss_mturk.pred = iss_pred.xs('both', level='feature'
                                    # ).drop(columns=['o_mention']
                                    # ).stack(
                                    # ).loc[iss_mturk.index]
            # iss_mturk.to_csv(join(MTURK_DIR, 'issue_mturk.csv'))
            print("Done!")
        
        ###################
        ## ad negativity ##
        ###################
        
        if an_flag:
        
            print("Classifying ad negativity...")
            
            # subset to train/test sample for ad negativity
            feat_neg = feat.loc[tone_wmp.index]
            
            # music feature columns
            d = 452
            mfeat_names = ['v' + str(e) for e in range(d)]
            
            # create text feature, tokenize so we don't repeat during training
            tfeat = [tokenize(transcript) if not pd.isna(transcript) else [] 
                     for transcript in feat_neg.transcript]
            
            # get audio features
            mfeat = feat_neg.loc[:, mfeat_names]
            
            # construct dataframe
            neg_feats = pd.DataFrame(mfeat, index=tone_wmp.index).assign(tfeat=tfeat)
            
            # train/test splits
            x_train,x_test,y_train,y_test = train_test_split(neg_feats,
                                                             tone_wmp,
                                                             test_size=0.2,
                                                             random_state=SEED)
            
            # results data frame
            neg_pred = pd.DataFrame(0, columns=['train', 'tone'],
                index=pd.MultiIndex.from_product([tone_wmp.index, ['text', 'music', 'both'],
                                                  ['lsvm', 'nsvm', 'knn', 'rf', 'nb']],
                                                 names=['creative', 'feature', 'model'])
                )
            neg_pred.loc[x_train.index, "train"] = 1
            
            ################
            ## Linear SVM ##
            ################
            
            print("\tFitting linear SVM classifiers... ")
    
            ## text only ##
    
            # pipeline
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                            input='content'), -1)],
                                    remainder='drop')),
                        ('dim_red', SelectPercentile(mutual_info_classif)),
                        ('clf', LinearSVC(loss='hinge', class_weight='balanced'))
                            ])
    
            # parameter grid for grid search
            params = [{
                       'dim_red__percentile' : [50, 75, 90, 100],
                       'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
                       'clf__class_weight': ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            lsvm_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            lsvm_t.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'text', 'lsvm'), 'tone'] = lsvm_t.predict(neg_feats)
    
            ## music only ##
    
            # pipeline    
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("ss", StandardScaler(), slice(-1))],
                                    remainder='drop')),
                        ('clf', LinearSVC(loss='hinge', class_weight='balanced'))
                            ])
    
            # parameter grid for grid search
            params = [{
                       'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
                       'clf__class_weight': ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            lsvm_m = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            lsvm_m.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'music', 'lsvm'), 'tone'] = lsvm_m.predict(neg_feats)
    
    
            ## text + music ##
                
            # pipeline    
            pipe = Pipeline([
                        ('feat1', ColumnTransformer(
                                        [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                                input='content'), -1)],
                                        remainder='passthrough')),
                        # music features run from -d:
                        ('feat2', ColumnTransformer(
                                    [("dim_red", SelectPercentile(mutual_info_classif), 
                                      slice(-d))], remainder='passthrough')),
                        # SVM inputs should be standardized
                        ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                                       accept_sparse=True)),
                        ('scaler', StandardScaler()),
                        ('clf', LinearSVC(loss='hinge', class_weight='balanced'))
                            ])
    
            # parameter grid for grid search
            params = [{
                       'feat2__dim_red__percentile' : [50, 75, 90, 100],
                       'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
                       'clf__class_weight': ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            lsvm_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            lsvm_tm.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'both', 'lsvm'), 'tone'] = lsvm_tm.predict(neg_feats)
            
            ###################
            ## Nonlinear SVM ##
            ###################
            
            print("\tFitting non-linear SVM classifiers... ")
            
            ## text only ##
    
            # pipeline
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                            input='content'), -1)],
                                    remainder='drop')),
                        ('dim_red', SelectPercentile(mutual_info_classif)),
                        ('clf', SVC(kernel='rbf', class_weight='balanced'))
                            ])
    
            # parameter grid for grid search
            params = [{
                       'dim_red__percentile' : [50, 75, 90, 100],
                       'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
                       'clf__gamma': np.logspace(-6, -2, 5),
                       'clf__class_weight': ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            svm_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            svm_t.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'text', 'nsvm'), 'tone'] = svm_t.predict(neg_feats)
    
            ## music only ##
    
            # pipeline    
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("ss", StandardScaler(), slice(-1))],
                                    remainder='drop')),
                        ('clf', SVC(kernel='rbf', class_weight='balanced'))
                            ])
    
            # parameter grid for grid search
            params = [{
                       'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
                       'clf__gamma': np.logspace(-6, -2, 5),
                       'clf__class_weight': ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            svm_m = GridSearchCV(pipe,params,scoring='accuracy',cv=5, verbose=1)
            svm_m.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'music', 'nsvm'), 'tone'] = svm_m.predict(neg_feats)
    
            ## text + music ##
                
            # pipeline    
            pipe = Pipeline([
                        ('feat1', ColumnTransformer(
                                        [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                                input='content'), -1)],
                                        remainder='passthrough')),
                        # music features run from -d:
                        ('feat2', ColumnTransformer(
                                    [("dim_red", SelectPercentile(mutual_info_classif), 
                                      slice(-d))], remainder='passthrough')),
                        # SVM inputs should be standardized
                        ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                                       accept_sparse=True)),
                        ('scaler', StandardScaler()),
                        ('clf', SVC(kernel='rbf', class_weight='balanced'))
                            ])
    
            # parameter grid for grid search
            params = [{
                       'feat2__dim_red__percentile' : [50, 75, 90, 100],
                       'clf__C': [0.001, 0.01, 0.1, 1, 5, 10],
                       'clf__gamma': np.logspace(-6, -2, 5),
                       'clf__class_weight': ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            svm_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            svm_tm.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'both', 'nsvm'), 'tone'] = svm_tm.predict(neg_feats) 
            
            #########
            ## KNN ##
            #########
            
            print("\tFitting KNN classifiers... ")
                
            ## text only ##
    
            # pipeline
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                            input='content'), -1)],
                                    remainder='drop')),
                        ('dim_red', SelectPercentile(mutual_info_classif)),
                        ('clf', KNeighborsClassifier())
                            ])
    
            # parameter grid for grid search
            params = [{
                      'dim_red__percentile' : [50, 75, 90, 100],
                      'clf__n_neighbors': [5, 7, 11, 15, 21]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            knn_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            knn_t.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'text', 'knn'), 'tone'] = knn_t.predict(neg_feats)
    
            ## music only ##
    
            # pipeline    
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("ss", StandardScaler(), slice(-1))],
                                    remainder='drop')),
                        ('clf', KNeighborsClassifier())
                            ])
    
            # parameter grid for grid search
            params = [{
                      'clf__n_neighbors': [5, 7, 11, 15, 21]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            knn_m = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            knn_m.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'music', 'knn'), 'tone'] = knn_m.predict(neg_feats)
    
            ## text + music ##
                
            # pipeline    
            pipe = Pipeline([
                        ('feat1', ColumnTransformer(
                                        [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                                input='content'), -1)],
                                        remainder='passthrough')),
                        # music features run from -d:
                        ('feat2', ColumnTransformer(
                                    [("dim_red", SelectPercentile(mutual_info_classif), 
                                      slice(-d))], remainder='passthrough')),
                        # KNN inputs should be standardized
                        ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                                       accept_sparse=True)),
                        ('scaler', StandardScaler()),
                        ('clf', KNeighborsClassifier())
                            ])
    
            # parameter grid for grid search
            params = [{
                      'feat2__dim_red__percentile' : [50, 75, 90, 100],
                      'clf__n_neighbors': [5, 7, 11, 15, 21]
                     }]
            
            # grid search
            print("\t\t", end="", flush=True)
            knn_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            knn_tm.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'both', 'knn'), 'tone'] = knn_tm.predict(neg_feats)
            
            ###################
            ## Random Forest ##
            ###################
            
            print("\tFitting random forest classifiers... ")
                
            ## text only ##
    
            # pipeline
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                            input='content'), -1)],
                                    remainder='drop')),
                        ('dim_red', SelectPercentile(mutual_info_classif)),
                        ('clf', RandomForestClassifier(class_weight='balanced', 
                                                       random_state=2002))
                            ])
    
            # parameter grid for grid search
            params = [{
                      'dim_red__percentile' : [50, 75, 90, 100],
                      'clf__n_estimators': [100, 250, 500],
                      'clf__min_samples_leaf': [1, 2, 4],
                      'clf__min_samples_split': [2, 5, 10],
                      'clf__class_weight' : ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            rf_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            rf_t.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'text', 'rf'), 'tone'] = rf_t.predict(neg_feats)
    
            ## music only ##
    
            # pipeline    
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [('ss', 'passthrough', slice(-1))], 
                                    remainder='drop')),
                        ('clf', RandomForestClassifier(class_weight='balanced',
                                                       random_state=2002))
                            ])
    
            # parameter grid for grid search
            params = [{
                      'clf__n_estimators': [100, 250, 500],
                      'clf__min_samples_leaf': [1, 2, 4],
                      'clf__min_samples_split': [2, 5, 10],
                      'clf__class_weight' : ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            rf_m = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            rf_m.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'music', 'rf'), 'tone'] = rf_m.predict(neg_feats)
    
            ## text + music ##
                
            # pipeline    
            pipe = Pipeline([
                        ('feat1', ColumnTransformer(
                                        [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                                input='content'), -1)],
                                        remainder='passthrough')),
                        # music features run from -d:
                        ('feat2', ColumnTransformer(
                                    [("dim_red", SelectPercentile(mutual_info_classif), 
                                      slice(-d))], remainder='passthrough')),
                        ('clf', RandomForestClassifier(class_weight='balanced',
                                                       random_state=2002))
                            ])
    
            # parameter grid for grid search
            params = [{
                      'feat2__dim_red__percentile' : [50, 75, 90, 100],
                      'clf__n_estimators': [100, 250, 500],
                      'clf__min_samples_leaf': [1, 2, 4],
                      'clf__min_samples_split': [2, 5, 10],
                      'clf__class_weight' : ['balanced', None]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            rf_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            rf_tm.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'both', 'rf'), 'tone'] = rf_tm.predict(neg_feats)
            
            #################    
            ## Naive Bayes ##
            #################
            
            print("\tFitting naive Bayes classifiers... ")
                
            ## text only ##
    
            # pipeline
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                            input='content'), -1)],
                                    remainder='drop')),
                        ('dim_red', SelectPercentile(mutual_info_classif)),
                        ('clf', MultinomialNB())
                            ])
    
            # parameter grid for grid search
            params = [{
                      'feat__cv' : [CountVectorizer(analyzer=dummy, min_df=2,
                                                    input='content')],
                      'dim_red__percentile': [50, 75, 90, 100],
                      'clf': [BernoulliNB(), MultinomialNB()],
                      'clf__alpha': [0.01, 0.1, 1, 2] 
                     },
                     {
                      'feat__cv' : [TfidfVectorizer(analyzer=dummy, min_df=2,
                                                    input='content')],
                      'dim_red__percentile': [50, 75, 90, 100],
                      'clf': [MultinomialNB()],
                      'clf__alpha': [0.01, 0.1, 1, 2] 
                     },
                     {
                      'feat__cv' : [TfidfVectorizer(analyzer=dummy, min_df=2,
                                                    input='content')],
                      'dim_red__percentile': [50, 75, 90, 100],
                      'clf': [GaussianNB()]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            nb_t = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            nb_t.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'text', 'nb'), 'tone'] = nb_t.predict(neg_feats)
    
            # music only
                
            # pipeline    
            pipe = Pipeline([
                        ('feat', ColumnTransformer(
                                    [('ss', 'passthrough', slice(-1))], 
                                    remainder='drop')),
                        ('clf', GaussianNB())
                            ])
    
            # fit classifier (no parameters to tune)
            print("\t\tFitting 1 fold for each of 1 candidate, totalling 1 fit", flush=True)
            nb_m = pipe
            nb_m.fit(x_train, y_train)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'music', 'nb'), 'tone'] = nb_m.predict(neg_feats)
            
            ## text + music ##
                
            # pipeline    
            pipe = Pipeline([
                        ('feat1', ColumnTransformer(
                                        [("cv", TfidfVectorizer(analyzer=dummy, min_df=2,
                                                                input='content'), -1)],
                                        remainder='passthrough')),
                        # music features run from -d:
                        ('feat2', ColumnTransformer(
                                    [("dim_red", SelectPercentile(mutual_info_classif), 
                                      slice(-d))], remainder='passthrough')),
                        # make output array dense
                        ('denser', FunctionTransformer(lambda x: x.toarray(), 
                                                       accept_sparse=True)),
                        ('clf', HeterogenousNB())
                            ])
    
            # parameter grid for grid search
            params = [{
                      'feat1__cv' : [CountVectorizer(analyzer=dummy, min_df=2,
                                                    input='content')],
                      'feat2__dim_red__percentile': [50, 75, 90, 100],
                      'clf__discrete_clf': ['bernoulli', 'multinomial'],
                      'clf__alpha': [0.01, 0.1, 1, 2] 
                     },
                     {
                      'feat1__cv' : [TfidfVectorizer(analyzer=dummy, min_df=2,
                                                    input='content')],
                      'feat2__dim_red__percentile': [50, 75, 90, 100],
                      'clf__discrete_clf': ['multinomial'],
                      'clf__alpha': [0.01, 0.1, 1, 2] 
                     },
                     {
                      'feat1__cv' : [TfidfVectorizer(analyzer=dummy, min_df=2,
                                                    input='content')],
                      'feat2__dim_red__percentile': [50, 75, 90, 100],
                      'clf': [GaussianNB()]
                     }]
    
            # grid search
            print("\t\t", end="", flush=True)
            nb_tm = GridSearchCV(pipe, params, scoring='accuracy', cv=5, verbose=1)
            nb_tm.fit(x_train, y_train, clf__split_ind=-d)
    
            # predictions
            neg_pred.loc[(neg_feats.index, 'both', 'nb'), 'tone'] = nb_tm.predict(neg_feats)
            print("Done!")
            
            # insert column for YouTube IDs
            uids_neg = [MATCHES_CMAG[ele] for ele in neg_pred.index.get_level_values('creative')]
            neg_pred = neg_pred.reset_index()
            neg_pred.insert(1, 'uid', uids_neg) # after creative
            
            # save results
            neg_pred.to_csv(join(ROOT, 'data', 'negativity_results.csv'), 
                            index=False)
    
    # read in data
    iss_pred = pd.read_csv(join(ROOT, 'data', 'mentions_results.csv'),
                          index_col=['creative', 'feature'])
    
    ## results ##
    
    print("Summarizing results... ", end='', flush=True)
    # issues
    iss_text = iss_pred.xs('text', level='feature').drop(columns=['uid', 'o_mention'])
    iss_both = iss_pred.xs('both', level='feature').drop(columns=['uid', 'o_mention'])
    
    # opponent mentions
    oment_text = iss_pred.xs('text', level='feature').o_mention
    oment_both = iss_pred.xs('both', level='feature').o_mention
    
    # confusion matrices
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    # issues
    cm_iss_text = pd.DataFrame(
                        confusion_matrix(iss_wmp.values.ravel(), 
                                         iss_text.values.ravel(), 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    cm_iss_both = pd.DataFrame(
                        confusion_matrix(iss_wmp.values.ravel(), 
                                         iss_both.values.ravel(), 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    # opponent mentions
    cm_oment_text = pd.DataFrame(
                        confusion_matrix(oment_wmp, 
                                         oment_text.loc[oment_wmp.index], 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    cm_oment_both = pd.DataFrame(
                        confusion_matrix(oment_wmp, 
                                         oment_both.loc[oment_wmp.index], 
                                         normalize='all'),
                        columns=cols, index=rows
                        )
    
    ## issue mention stats
    n_iss = iss_wmp.size
    cm_iss_text_un =  (cm_iss_text * n_iss).astype(int)
    cm_iss_both_un =  (cm_iss_both * n_iss).astype(int)
    
    pos_inc = (cm_iss_both_un[('Auto', 'Yes')].sum() - 
               cm_iss_text_un[('Auto', 'Yes')].sum())
    fp_delta = (cm_iss_both_un.loc[('WMP', 'No'), ('Auto', 'Yes')] - 
                cm_iss_text_un.loc[('WMP', 'No'), ('Auto', 'Yes')])
    tp_delta = (cm_iss_both_un.loc[('WMP', 'Yes'), ('Auto', 'Yes')] - 
                cm_iss_text_un.loc[('WMP', 'Yes'), ('Auto', 'Yes')])
    
    fpr_text = (cm_iss_text_un.loc[('WMP', 'No'), ('Auto', 'Yes')] / 
                (cm_iss_text_un.loc[('WMP', 'No'), ('Auto', 'No')] +
                 cm_iss_text_un.loc[('WMP', 'No'), ('Auto', 'Yes')])
               )
    fnr_text = (cm_iss_text_un.loc[('WMP', 'Yes'), ('Auto', 'No')] / 
                (cm_iss_text_un.loc[('WMP', 'Yes'), ('Auto', 'Yes')] +
                 cm_iss_text_un.loc[('WMP', 'Yes'), ('Auto', 'No')])
               )
    
    fpr_both = (cm_iss_both_un.loc[('WMP', 'No'), ('Auto', 'Yes')] / 
                (cm_iss_both_un.loc[('WMP', 'No'), ('Auto', 'No')] +
                 cm_iss_both_un.loc[('WMP', 'No'), ('Auto', 'Yes')])
               )
    fnr_both = (cm_iss_both_un.loc[('WMP', 'Yes'), ('Auto', 'No')] / 
                (cm_iss_both_un.loc[('WMP', 'Yes'), ('Auto', 'Yes')] +
                 cm_iss_both_un.loc[('WMP', 'Yes'), ('Auto', 'No')])
               )
    
    # issue validation study
    iss_val = pd.read_csv(join(VAL_DIR, 'issue_validation.csv'), 
                          index_col='creative')
    
    # number of false positives/negatives
    nfp = (iss_val.pred == 1).sum()
    nfn = iss_val.shape[0] - nfp
    
    # number of mistakes by wmp
    nmis_fp = ((iss_val.pred == 1) & (iss_val.note == 'wmp wrong')).sum()
    nmis_fn = ((iss_val.pred == 0) & (iss_val.note == 'wmp wrong')).sum()
    
    # mturk issue validation stats
    iss_mturk = pd.read_csv(join(MTURK_DIR, 'issue_mturk.csv'), 
                            index_col=['creative', 'issue'])
    
    # convert columns to index for wmp data
    iss_stacked = iss_wmp.stack()
    iss_stacked.index.set_names(['creative', 'issue'], inplace=True)
    # subset to mturk sample
    iss_wmp_mt = iss_stacked.loc[iss_mturk.index]
    
    # mistakes / true
    n_mturk = iss_mturk.shape[0] // 5
    nerr_mturk = (iss_mturk.pred != iss_wmp_mt).sum() // 5
    nagr_mturk = n_mturk - nerr_mturk
    
    # false positives/negatives
    nfp_mturk = ((iss_mturk.pred == 1) & (iss_wmp_mt == 0)).sum() // 5
    nfn_mturk = nerr_mturk - nfp_mturk
    
    # output
    with open(join(ROOT, 'results', 'performance', 'issue_results.txt'), 'w') as fh:
        print("Issue Mention Results", file=fh)
        print("---------------------", file=fh)
        print(file=fh)
        print("Video-Issue Pairs: {}".format(n_iss), file=fh)
        print(file=fh)
        print("Increase in positive samples w/ "
              "images: {} ({:.2%})".format(pos_inc, pos_inc/n_iss) ,file=fh)
        print("Increase in false positive samples w/ "
              "images: {} ({:.0%})".format(fp_delta, fp_delta/pos_inc) ,file=fh)
        print("Increase in true positive samples w/ "
              "images: {} ({:.0%})".format(tp_delta, tp_delta/pos_inc) ,file=fh)
        print(file=fh)
        print("FPR (Text): {:.2%}".format(fpr_text), file=fh)
        print("FNR (Text): {:.0%}".format(fnr_text), file=fh)
        print("FPR (Both): {:.2%}".format(fpr_both), file=fh)
        print("FNR (Both): {:.0%}".format(fnr_both), file=fh)
        print(file=fh)
        print("Issue Mention Validation Study", file=fh)
        print("------------------------------", file=fh)
        print("# of false positives: {}".format(nfp), file=fh)
        print("# of false negatives: {}".format(nfn), file=fh)
        print(file=fh)
        print("# mistakes made by WMP for FP: {}".format(nmis_fp), file=fh)
        print("# mistakes made by us for FP: {}".format(nfp - nmis_fp), file=fh)
        print("# mistakes made by WMP for FN: {}".format(nmis_fn), file=fh)
        print("# mistakes made by us for FN: {}".format(nfn - nmis_fn), file=fh)
        print(file=fh)
        print("Issue Mention MTurk Study", file=fh)
        print("------------------------------", file=fh)
        print("# of samples: {}".format(n_mturk), file=fh)
        print("# of samples in agreement: {}".format(nagr_mturk), file=fh)
        print("# of samples in disagreement: {}".format(nerr_mturk), file=fh)
        print("# of false positives: {}".format(nfp_mturk), file=fh)
        print("# of false negatives: {}".format(nfn_mturk), file=fh)
        
    
    ## opponent mention stats
    oment_val = pd.read_csv(join(VAL_DIR, 'oppment_validation.csv'), 
                            index_col='creative')
    
    # number of mistakes
    nmis_om = oment_val.shape[0]
    
    # fp/fn
    nfp = (oment_val.pred == 1).sum()
    nfn = nmis_om - nfp
    
    # number of fp/fn samples that wmp mislabeled
    nmis_fp = ((oment_val.pred == 1) & oment_val.note.isin(['wmp wrong', 'image text'])).sum()
    nmis_fn = ((oment_val.pred == 0) & (oment_val.note == 'wmp wrong')).sum()
    
    # total number of mistakes by wmp
    nmis_wmp = nmis_fp + nmis_fn
    
    # mistakes by us due to transcript
    nmis_trans = oment_val.note.str.contains('transcript').sum()
    
    # positive sample changes
    n_om = oment_wmp.size
    cm_oment_text_un =  (cm_oment_text * n_om).astype(int)
    cm_oment_both_un =  (cm_oment_both * n_om).astype(int)
    
    pos_inc = (cm_oment_both_un[('Auto', 'Yes')].sum() - 
               cm_oment_text_un[('Auto', 'Yes')].sum())
    fp_delta = (cm_oment_both_un.loc[('WMP', 'No'), ('Auto', 'Yes')] - 
                cm_oment_text_un.loc[('WMP', 'No'), ('Auto', 'Yes')])
    tp_delta = (cm_oment_both_un.loc[('WMP', 'Yes'), ('Auto', 'Yes')] - 
                cm_oment_text_un.loc[('WMP', 'Yes'), ('Auto', 'Yes')])
    
    with open(join(ROOT, 'results', 'performance', 'oppment_results.txt'), 'w') as fh:
        print("Opponent Mention Results", file=fh)
        print("------------------------", file=fh)
        print("Total # of videos: {}".format(oment_wmp.shape[0]), file=fh)
        print("# of disagreements: {} ({:.0%})".format(nmis_om, nmis_om / n_om), file=fh)
        print("Increase in positive samples w/ "
              "images: {} ({:.0%})".format(pos_inc, pos_inc/n_om) ,file=fh)
        print("Increase in false positive samples w/ "
              "images: {} ({:.0%})".format(fp_delta, fp_delta/pos_inc) ,file=fh)
        print("Increase in true positive samples w/ "
              "images: {} ({:.0%})".format(tp_delta, tp_delta/pos_inc) ,file=fh)
        print(file=fh)
        print("Opponent Mention Validation Study", file=fh)
        print("---------------------------------", file=fh)
        print("# of false positives: {}".format(nfp), file=fh)
        print("# of false negatives: {}".format(nfn), file=fh)
        print(file=fh)
        print("# mistakes made by WMP for FP: {}".format(nmis_fp), file=fh)
        print("# mistakes made by us for FP: {}".format(nfp - nmis_fp), file=fh)
        print("# mistakes made by WMP for FN: {}".format(nmis_fn), file=fh)
        print("# mistakes made by us for FN: {}".format(nfn - nmis_fn), file=fh)
        print(file=fh)
        print("# of FN mistakes made by us due to transcript: {}".format(nmis_trans), file=fh)
        print(file=fh)
        print("Overall accuracy accounting for WMP mistakes: {} ({:.0%})".format(nmis_wmp, nmis_wmp/nmis_om), file=fh)

    print("Done!")
if __name__ == '__main__':
    main()