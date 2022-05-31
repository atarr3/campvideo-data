import numpy as np
import pandas as pd

from os.path import abspath, dirname, join
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import LinearSVC

# root folder for replication repo
ROOT = dirname(dirname(abspath(__file__)))

# seed for replication
SEED = 2002

# wmp/cmag data
WMP_DIR = join(ROOT, 'data', 'wmp')

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
    
    # confusion matrices
    pd.options.display.float_format = '{:,.2%}'.format
    cols = pd.MultiIndex.from_tuples([('Auto', 'No'), ('Auto', 'Yes')])
    rows = pd.MultiIndex.from_tuples([('WMP', 'No'), ('WMP', 'Yes')])
    
    cm_o = pd.DataFrame(
                confusion_matrix(o_wmp, o_pred, normalize='all'),
                columns=cols, index=rows
                )
    cm_f = pd.DataFrame(
                confusion_matrix(f_wmp, f_pred, normalize='all'),
                columns=cols, index=rows
                )
    
    # confusion matrix (uncorrected, Table 4)
    with open(join(ROOT, 'tables', 'table4.txt'), 'w') as fh:
        print("Favored Candidate", file=fh)
        print("-----------------", file=fh)
        print(cm_f, file=fh)
        print(file=fh)
        print("Opposing Candidate", file=fh)
        print("------------------", file=fh)
        print(cm_o, file=fh)
        
if __name__ == '__main__':
    main()