import xtools as xt
import numpy as np
import time
import xfeatures as xf

start = time.time()
df_train = xt.read_pickle(path = "data/processed", sample = "train")
df_test = xt.read_pickle(path = "data/processed", sample = "test")
#df_train.drop(columns='isFraud', axis=1, inplace=True)
end = time.time()

print("Read off data: %.1f sec"%(end-start)) 
print("Data train shape", df_train.shape)
print("Data test shape", df_test.shape)
print(f"Train columns {df_train.columns.values}")
print(f"Test columns {df_test.columns.values}")

# data cleaning ###############################################################################

# Cleaning infinite values to NaN
# by https://www.kaggle.com/dimartinot

#start = time.time()
#df_train.replace([np.inf, -np.inf], np.nan, inplace=True)

#df_train = clean_inf_nan(df_train)
#df_test = clean_inf_nan(df_test)

#df_train.fillna(-1,inplace=True)
#df_test.fillna(-1,inplace=True)

#end = time.time()

#print("Cleaning data: %.1f sec"%(end-start)) 
#print(df_train.shape)
#print(df_test.shape)

# map emails ###############################################################################
def steer_emails(df = None):

    emails = {'gmail': 'google',
              'att.net': 'att',
              'twc.com': 'spectrum', 
              'scranton.edu': 'other',
              'optonline.net': 'other',
              'hotmail.co.uk': 'microsoft',
              'comcast.net': 'other',
              'yahoo.com.mx': 'yahoo',
              'yahoo.fr': 'yahoo',
              'yahoo.es': 'yahoo',
              'charter.net': 'spectrum',
              'live.com': 'microsoft', 
              'aim.com': 'aol',
              'hotmail.de':'microsoft',
              'centurylink.net': 'centurylink',
              'gmail.com': 'google',
              'me.com': 'apple',
              'earthlink.net': 'other',
              'gmx.de': 'other',
              'web.de': 'other',
              'cfl.rr.com': 'other',
              'hotmail.com': 'microsoft', 
              'protonmail.com': 'other',
              'hotmail.fr': 'microsoft',
              'windstream.net': 'other', 
              'outlook.es': 'microsoft',
              'yahoo.co.jp': 'yahoo',
              'yahoo.de': 'yahoo',
              'servicios-ta.com':'other',
              'netzero.net': 'other',
              'suddenlink.net': 'other',
              'roadrunner.com': 'other',
              'sc.rr.com': 'other',
              'live.fr': 'microsoft',
              'verizon.net': 'yahoo',
              'msn.com': 'microsoft',
              'q.com': 'centurylink', 
              'prodigy.net.mx': 'att',
              'frontier.com': 'yahoo',
              'anonymous.com': 'other', 
              'rocketmail.com': 'yahoo',
              'sbcglobal.net': 'att',
              'frontiernet.net': 'yahoo', 
              'ymail.com': 'yahoo',
              'outlook.com': 'microsoft',
              'mail.com': 'other', 
              'bellsouth.net': 'other',
              'embarqmail.com': 'centurylink',
              'cableone.net': 'other', 
              'hotmail.es': 'microsoft',
              'mac.com': 'apple',
              'yahoo.co.uk': 'yahoo',
              'netzero.com': 'other', 
              'yahoo.com': 'yahoo',
              'live.com.mx': 'microsoft',
              'ptd.net': 'other',
              'cox.net': 'other',
              'aol.com': 'aol',
              'juno.com': 'other',
              'icloud.com': 'apple'}

    us_emails = ['gmail', 'net', 'edu']

    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
    for c in ['P_emaildomain', 'R_emaildomain']:
        df_train[c + '_bin']    = df_train[c].map(emails)
        df_test[c + '_bin']     = df_test[c].map(emails)
   
        df_train[c + '_suffix'] = df_train[c].map(lambda x: str(x).split('.')[-1])
        df_test[c + '_suffix']  = df_test[c].map(lambda x: str(x).split('.')[-1])
    
        df_train[c + '_suffix'] = df_train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
        df_test[c + '_suffix']  = df_test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

        #xf.features.append('P_emaildomain_bin')
        #xf.features.append('R_emaildomain_bin')
        
start = time.time()
steer_emails(df = df_merge)
end = time.time()

print("Email mapping: %.1f sec"%(end-start))    
print(df_train.shape)
print(df_test.shape)
print(f"Train columns {df_train.columns.values}")
print(f"Test columns {df_test.columns.values}")
print(df_train['P_emaildomain_bin'].head())
print(df_train['P_emaildomain_suffix'].head())


# TimeDelta Feature ########################################################################
def transaction_dt():
    # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
    df_test['Transaction_dow'] = np.floor((df_test['TransactionDT'] / (3600 * 24) - 1) % 7) #Creates a day of the week feature, encoded as 0-6. 
    df_test['Transaction_hour'] = np.floor(df_test['TransactionDT'] / 3600) % 24  # Creates an hour of the day feature, encoded as 0-23. 

    df_train['Transaction_dow'] = np.floor((df_train['TransactionDT'] / (3600 * 24) - 1) % 7)
    df_train['Transaction_hour'] = np.floor(df_train['TransactionDT'] / 3600) % 24

    #transaction_data_columns = test_transaction_data.columns
    #numericCols = test_transaction_data._get_numeric_data().columns
    #categoricalCols = list(set(transaction_data_columns) - set(numericCols))
    #test_transaction_data[categoricalCols] = test_transaction_data[categoricalCols].replace({ np.nan:'missing'})
    #test_transaction_data[numericCols] = test_transaction_data[numericCols].replace({ np.nan:-1})

start = time.time()
transaction_dt()
end = time.time()
print("Transaction delta time: %.1f sec"%(end-start))
print(df_train.shape)
print(df_test.shape)

# TimeDelta Feature ########################################################################
def dT_trans(df = None):
    import datetime
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df["Date"]     = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    df['Weekdays'] = df['Date'].dt.dayofweek
    df['Hours']    = df['Date'].dt.hour
    df['Days']     = df['Date'].dt.day

    return df

start = time.time()
#df_train = dT_trans(df_train)
#df_test = dT_trans(df_test)
end = time.time()
print("Time delta: %.1f sec"%(end-start))
print(df_train.shape)
print(df_test.shape)


# Encoding categorical features & treat NaN #############################################################################
from sklearn import preprocessing
    
def categ_trans_nan():
    """
    Categorical Features - Transaction
    ProductCD
    card1 - card6
    addr1, addr2
    P_emaildomain
    R_emaildomain
    M1 - M9

    Categorical Features - Identity
    DeviceType
    DeviceInfo
    id_12 - id_38
    """

    train_columns             = df_train.columns
    train_numeric_columns     = df_train._get_numeric_data().columns
    train_categorical_columns = list(set(train_columns) - set(train_numeric_columns))

    test_columns             = df_test.columns
    test_numeric_columns     = df_test._get_numeric_data().columns
    test_categorical_columns = list(set(test_columns) - set(test_numeric_columns))

    print('Train set - categorical columns:', sorted(train_categorical_columns))

    # categorical fill NaN
    df_train[train_categorical_columns].replace({ np.nan:'missing'}, inplace = True)
    df_test[test_categorical_columns].replace({ np.nan:'missing'}, inplace = True)

    # categorical fill NaN
    df_train[train_numeric_columns].replace(np.nan, 0, inplace = True)
    df_test[test_numeric_columns].replace(np.nan, 0, inplace = True)

    print("Train categorical head")
    print(df_train[train_categorical_columns].head)

    print("Train numeric head")
    print(df_train[train_numeric_columns].head)

    #transform / label encoding
    for f in df_train.drop('isFraud', axis=1).columns:
        if df_train[f].dtype=='object' or df_test[f].dtype=='object':         
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df_train[f].values) + list(df_test[f].values))
            df_train[f] = lbl.transform(list(df_train[f].values))
            df_test[f] = lbl.transform(list(df_test[f].values))   

    #done

#
start = time.time()
categ_trans_nan()
end = time.time()

print("Encoding categorical features & fill NaNs: %.1f sec"%(end-start))
print(df_train.shape)
print(df_test.shape)


# fill NaN  ###############################################################################
start = time.time()

#df_train.fillna(-1, inplace=True)
#df_test.fillna(-1, inplace=True)

#df_train.fillna(value=df_train.mean(), inplace=True)

end = time.time()
print("Fill NaNs: %.1f sec"%(end-start))
print(df_train.shape)
print(df_test.shape)


# Set features & target ########################################################################

#target
target = 'isFraud'

#features
features = [ "P_emaildomain_bin",
             "R_emaildomain_bin",
             "Transaction_hour"
]

features += xf.features

print(f"Features {features}")


########################################################
#set X & y
########################################################

print("Train shape", df_train.shape)
print("Test shape", df_test.shape)

if target not in df_train.columns:
    print("Target column not existing in train dataset...")
    exit(1)
else:
    print(f"Found target column {target} in train dataset...")

X_train = df_train.sort_values('TransactionDT').drop([target, 
                                                      'TransactionDT'],
                                                     axis=1)[features]
    
y_train = df_train.sort_values('TransactionDT')[target].astype(bool)

X_test = df_test.sort_values('TransactionDT').drop(['TransactionDT'], 
                                                   axis=1)[features]

#X_train, y_train = df_train[features], df_train[target].astype(bool)
#X_test, y_test = df_test[features], df_test[target].astype(bool)

#df_test = df_test[['TransactionID', target]]   
#y_pred = np.zeros(len(df_test))

print("X train shape", X_train.shape)
print("y train shape", y_train.shape)

#https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
print("X train has NaN", X_train.isnull().any().any())
print("y train has NaN", y_train.isnull().any().any())


########################################################
#algo
########################################################

# SKL
from sklearn.ensemble import RandomForestClassifier

rf_act = RandomForestClassifier(
    n_estimators=50, #50
    #criterion='gini',
    max_depth=5, #5
    min_samples_split=2,
    #min_samples_leaf=1,
    #min_weight_fraction_leaf=0.0,
    #max_features='auto',
    #max_leaf_nodes=None,
    #min_impurity_decrease=0.0,
    #min_impurity_split=None,
    #bootstrap=True,
    #oob_score=False,
    #n_jobs=-1,
    #random_state=0,
    #verbose=0,
    #warm_start=False,
    #class_weight='balanced'
)

rf_opt = RandomForestClassifier(random_state = 0)


clf_skl_act = rf_act
clf_skl_opt = rf_opt

# Light GBM
import lightgbm as lgb


lgb_params = {'num_leaves': 256,
              'min_child_samples': 10, #79
              'objective': 'binary',
              'max_depth': 5, #14
              'learning_rate': 0.07, # 0.03
              "boosting_type": "gbdt",
              "subsample_freq": 3,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'auc',
              "verbosity": 1,
              #'reg_alpha': 0.3,
              #'reg_lambda': 0.3,
              #'colsample_bytree': 0.9,
              #'categorical_feature': cat_cols
}

n_estimators = 1000
n_jobs = -1

clf_lgb_act = lgb.LGBMClassifier(**lgb_params, n_estimators = n_estimators, n_jobs = n_jobs)

# - - - - - - - - - - - - - - - - - - - - - - - - - - -
do_make_predictions_skl = False
do_make_predictions_lgb = True

########################################################
# predictions
########################################################
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy import interp
import matplotlib.pyplot as plt
import gc
def plot_roc_curve(fprs, tprs):
    """Plot the Receiver Operating Characteristic from a list
    of true positive rates and false positive rates.
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    
    # Initialize useful lists + the plot axes.
    tprs_interp = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    f, ax = plt.subplots(figsize=(14,10))
    
    # Plot ROC for each K-Fold + compute AUC scores.
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
        tprs_interp[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
     
    # Plot the luck line.
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    
    # Plot the mean ROC.
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)
    
    # Plot the standard deviation around the mean ROC.
    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    # Fine tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)

#from numba import jit
#
#@jit
def fast_auc(y_true, y_prob):
    """
    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc


def eval_auc(y_true, y_pred):
    """
    Fast auc eval function for lgb.
    """
    return 'auc', fast_auc(y_true, y_pred), True

def make_predictions_lgb(X_train = None,
                         y_train = None,
                         nfolds = 5,
                         algo_act = None,
                         verbose = 500,
                         early_stopping_rounds = 10,
                         stratify = True,
                         seed = 123,
                         shuffle = True,
                         eval_metric = "auc"):

    time_beg = time.time()
    
    if stratify:
        folds = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)
    else:
        folds = KFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)



    predictors = X_train.columns.values.tolist()[2:]

    for ifold, (tr_idx, vl_idx) in enumerate(folds.split(X_train, y_train)):

        print("Fold:", ifold)
        print ("Train idx", tr_idx)
        print ("Valid idx", vl_idx)

        # use iloc instead of X_train[tr_idx]
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[vl_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        print("X train size %i, X validation size %i"%(len(X_tr),len(X_vl)))

        #fit
        print('Start fitting estimator')
        gc.collect()
        print('First gc done')
        gc.collect()
        print('Second gc done')

        #algo_act.fit(X_train,
        #             y_train, 
        #             eval_set=[(X_tr, y_tr), (X_vl, y_vl)],
        #             #eval_metric=eval_auc if eval_metric is "auc" else None,
        #             verbose=verbose,
        #             early_stopping_rounds=early_stopping_rounds)

        #predictions
        #y_pred_valid = algo_act.predict_proba(X_vl)[:, 1]
        #y_pred_test = algo_act.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]

        xg_train = lgb.Dataset(X_tr[predictors].values,
                               label=y_tr.values,
                               feature_name=predictors,
                               free_raw_data = False )

        xg_valid = lgb.Dataset(X_vl[predictors].values,
                               label=y_vl[target].values,
                               feature_name=predictors,
                               free_raw_data = False)   

    
        clf = algo_act.train(params, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=50, early_stopping_rounds = 50) 

    delta_time = time.time() - time_beg
    print(f"Total training time: {round(delta_time / 60, 2)}")


def make_predictions_skl(X_train = None,
                         y_train = None,
                         nfolds = 2,
                         stratify = False,
                         seed= 123,
                         shuffle = True,
                         algo_act = None,
                         algo_opt = None,
                         optimize = False):

    time_beg = time.time()
    
    if stratify:
        folds = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)
    else:
        folds = KFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)

    y_train_predictions = np.zeros(X_test.shape[0])
    y_train_oof = np.zeros(X_train.shape[0])
    list_roc_auc_score = []
    list_tpr = []
    list_fpr = []
    list_auc = []
    for ifold, (tr_idx, vl_idx) in enumerate(folds.split(X_train, y_train)):

        print("Fold:", ifold)
        print ("Train X idx", tr_idx)
        print ("Train y idx", vl_idx)

        # use iloc instead of X_train[tr_idx]
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[vl_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        print("X train size %i, X validation size %i"%(len(X_tr),len(X_vl)))

        #opt on first fold
        if optimize and ifold == 0:
            param_grid = {'max_depth': range(2,4,1),
                          'min_samples_split': range(2, 4, 2),
                          'n_estimators': range(100, 401, 100)}
    
            scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

            clf = GridSearchCV(estimator = clf_opt,
                               param_grid = param_grid,
                               scoring = scoring,
                               refit='AUC',
                               return_train_score=True)
        
            clf.fit(X_tr, y_tr.values.ravel())
            print('Best score (AUC): ', clf.best_score_)
            print('Best hyperparameters (max AUC): ', clf.best_params_)
            print('Best parameters set:')
            best_parameters = clf.best_estimator_.get_params()
            for param_name in sorted(param_grid.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]) )

        #fit        
        algo_act.fit(X_tr, y_tr)
        
        #predicted prob
        y_pred_vl = algo_act.predict_proba(X_vl)[:,1]
        y_train_oof[vl_idx] = y_pred_vl
        
        #roc auc score
        score = make_scorer(roc_auc_score, needs_proba=True)(algo_act, X_vl, y_vl)
        print(f"Score {score}")

        auc_score = roc_auc_score(y_vl, y_pred_vl)
        list_roc_auc_score.append(auc_score)
        print(f"AUC {auc_score}")

        # Compute ROC curve and AUC: use validation sample 
        fpr, tpr, thresholds = roc_curve(y_vl, y_pred_vl)
        list_tpr.append(tpr)
        list_fpr.append(fpr)
              
        #release mem
        del X_tr, X_vl, y_tr, y_vl
        gc.collect()

    mean_roc_auc_score  = sum(list_roc_auc_score, 0.0)/len( list_roc_auc_score )
    y_train_predictions /= len( y_train_predictions )

    print(f"Mean AUC {mean_roc_auc_score}")

    # out-of-fold predictions on train data
    mean_roc_auf_oof = roc_auc_score(y_train, y_train_oof)
    print("OOF AUC = {}".format(mean_roc_auf_oof))

    delta_time = time.time() - time_beg

    print(f"Total training time: {round(delta_time / 60, 2)}")

    #plot ROC curve
    plot_roc_curve(fprs = list_fpr, tprs = list_tpr)    

# prediction ============================================

if do_make_predictions_skl:
    make_predictions_skl(X_train = X_train,
                         y_train = y_train,
                         nfolds = 5,
                         stratify = True,
                         algo_act = clf_skl_act,
                         algo_opt = clf_skl_opt,
                         optimize = False)

if do_make_predictions_lgb:
    make_predictions_lgb(X_train = X_train,
                         y_train = y_train,
                         nfolds = 5,
                         algo_act = clf_lgb_act,
                         verbose = 500,
                         early_stopping_rounds = 200,
                         stratify = True,
                         seed = 123,
                         shuffle = True,
                         eval_metric = "auc")

exit(1)
########################################################
#ROC
########################################################
# Use StratifiedKFold which guarantees that in each split we have a constant amount of samples from each class.
# https://www.kaggle.com/ynouri/random-forest-k-fold-cross-validation


def compute_roc_auc(index):
    y_predict = clf.predict_proba(X.iloc[index])[:,1]
    fpr, tpr, thresholds = roc_curve(y.iloc[index], y_predict)
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score



import pandas as pd

n_splits = 5

results = pd.DataFrame(columns=['training_score', 'test_score'])
fprs, tprs, scores = [], [], []

for (train, test), i in zip(cv.split(X, y), range(n_splits)):
    
    clf.fit(X.iloc[train], y.iloc[train])
    _, _, auc_score_train = compute_roc_auc(train)
    fpr, tpr, auc_score = compute_roc_auc(test)
    scores.append((auc_score_train, auc_score))
    fprs.append(fpr)
    tprs.append(tpr)

plot_roc_curve(fprs, tprs)
pd.DataFrame(scores, columns=['AUC Train', 'AUC Test'])
    
#from sklearn.metrics import roc_curve, auc #, accuracy_score, make_scorer, classification_report, confusion_matrix
#from sklearn.metrics import plot_roc_curve
#



