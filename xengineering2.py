import xtools as xt
import xfeatures as xf
import numpy as np
import pandas as pd
import seaborn as sns
import time
import statistics as st
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, make_scorer, roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from scipy import interp
import matplotlib.pyplot as plt
import gc
from xgboost import plot_importance
from itertools import cycle
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

# - - - - - - - - - - - - - - - - - - - - - - - - - - -
#LGBM
do_make_predictions_lgbm = False
lgbm_n_estimators = 200
lgbm_importance_threshold = 10
lgbm_filter_best_features = False
lgbm_use_preselected_best_features = True

#RF
do_make_predictions_skl = True
skl_n_estimators = 600
do_plots = True
do_sub = False
skl_optimize = False
skl_importance_threshold = 0.002
skl_filter_best_features = True
skl_use_preselected_best_features = False

#xgb
do_make_predictions_xgb = False
xgb_n_trees = 600

do_make_predictions_simple_xgb = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - -
pd.set_option('display.max_rows', None)
    
start = time.time()
df_train = xt.read_pickle(path = "data/processed", sample = "train")
df_test = xt.read_pickle(path = "data/processed", sample = "test")
#df_train.drop(columns='isFraud', axis=1, inplace=True)

if do_sub:
    df_sub =pd.read_csv('data/sample_submission.csv', index_col='TransactionID')
    print("Sub", df_sub.head)
    print("Sub",df_sub.shape)

    
end = time.time()

print("Read off data: %.1f sec"%(end-start)) 
print("Data train shape", df_train.shape)
print("Data test shape", df_test.shape)
print("Train columns", df_train.columns.values)
print("Test columns", df_test.columns.values)

# Set features & target ########################################################################
#target
target = 'isFraud'
features = xf.features

print(f"Features {features}")
print(f"Number of Features {len(features)}")

########################################################
#set X & y
########################################################

print("DF Train shape", df_train.shape)
print("DF Test shape", df_test.shape)

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
X_test.replace([np.inf, -np.inf], np.nan)

print("X train shape", X_train.shape)
print("y train shape", y_train.shape)

#https://stackoverflow.com/questions/29530232/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe
print("X train has NaN", X_train.isnull().any().any())
print("y train has NaN", y_train.isnull().any().any())

if do_sub:
    print("X test shape", X_test.shape)

########################################################
# feature scaling
########################################################
#https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
print("Info: Scaling data...")
sc = StandardScaler()
X_train[X_train.columns] = sc.fit_transform(X_train[X_train.columns])
X_test[X_test.columns] = sc.fit_transform(X_test[X_test.columns])

print("Info: X type", type(X_train))

########################################################
# SKL algo
########################################################
rf_act = RandomForestClassifier(
    n_estimators=skl_n_estimators, #100
    #criterion='gini',
    max_depth=10, #5
    min_samples_split=2,
    #min_samples_leaf=1,
    #min_weight_fraction_leaf=0.0,
    #max_features='auto',
    #max_leaf_nodes=None,
    #min_impurity_decrease=0.0,
    #min_impurity_split=None,
    #bootstrap=True,
    #oob_score=False,
    n_jobs=4,
    random_state=0,
    #verbose=0,
    #warm_start=False,
    #class_weight='balanced'
)

rf_opt = RandomForestClassifier(random_state = 0)

clf_skl_act = rf_act
clf_skl_opt = rf_opt

skl_best_variables_list = ['TransactionAmt','ProductCD','card1','card2','card3','card5','card6'
,'addr1','addr2','dist2','P_emaildomain','R_emaildomain','C1','C2','C4'
,'C6','C7','C8','C9','C10','C11','C12','C13','C14','D1','D2','D3','D4'
,'D5','D6','D7','D8','D9','D10','D12','D13','D14','D15','M4','V19','V20'
,'V22','V23','V24','V33','V34','V38','V40','V44','V45','V53','V54','V55'
,'V56','V57','V58','V61','V62','V66','V67','V70','V73','V74','V78','V83'
,'V86','V87','V93','V94','V95','V96','V97','V99','id_01','id_02','id_03'
,'id_05','id_06','id_09','id_10','id_12','id_13','id_17','id_19','id_20'
,'id_30','id_31','id_33','id_34','id_35','id_38','DeviceType','DeviceInfo'
,'Transaction_hour']

########################################################
# LGBM algo
########################################################

lgbm_params = {'num_leaves': 256,
               'min_child_samples': 79, #important: low values lead to crash
               'objective': 'binary',
               'max_depth': 13, #14
               'learning_rate': 0.03, # 0.03
               'objective': 'binary',
               "boosting_type": "gbdt",
               "subsample_freq": 3,
               "subsample": 0.9,
               "bagging_seed": 11,
               "metric": 'auc',
               "verbosity": 1,
               #"is_unbalance": True,
               #'reg_alpha': 0.3,
               #'reg_lambda': 0.3,
               #'colsample_bytree': 0.9,
               #'categorical_feature': cat_cols
               #'early_stopping_round' : 20, #100
               #'boost_from_average': False, #True
               'seed': 1337,
               'feature_fraction_seed': 1337,
               'bagging_seed': 1337,
               'drop_seed': 1337,
               'data_random_seed': 1337,
               "max_bin" : 63,
               "device_type" : "cpu",
               "num_threads" : 4,
}

clf_lgbm_act = lgbm.LGBMClassifier(**lgbm_params,
                                   n_estimators = lgbm_n_estimators)

lgbm_best_variables_list = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10', 'V11', 'V12', 'V13', 'V19', 'V20', 'V23', 'V24', 'V25', 'V26', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V64', 'V66', 'V67', 'V69', 'V70', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V85', 'V86', 'V87', 'V90', 'V91', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_24', 'id_25', 'id_26', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'Transaction_hour']

########################################################
# XGB algo
########################################################
def xgb_objective(params):
    time1 = time.time()
    params = {
        #'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'subsample': "{:.2f}".format(params['subsample']),
        'reg_alpha': "{:.3f}".format(params['reg_alpha']),
        'reg_lambda': "{:.3f}".format(params['reg_lambda']),
        'learning_rate': "{:.3f}".format(params['learning_rate']),
        'num_leaves': '{:.3f}'.format(params['num_leaves']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        'min_child_samples': '{:.3f}'.format(params['min_child_samples']),
        'feature_fraction': '{:.3f}'.format(params['feature_fraction']),
        'bagging_fraction': '{:.3f}'.format(params['bagging_fraction'])
    }

    print("\n############## New Run ################")
    print(f"params = {params}")
    FOLDS = 7
    count=1
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
    y_oof = np.zeros(X_train.shape[0])
    score_mean = 0
    for tr_idx, val_idx in skf.split(X_train, y_train):
        clf = xgb.XGBClassifier(
            n_estimators = 600,
            random_state=4, verbose=True, 
            tree_method='gpu_hist', 
            **params
        )

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)
        #y_pred_train = clf.predict_proba(X_vl)[:,1]
        #print(y_pred_train)
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)
        # plt.show()
        score_mean += score
        print(f'{count} CV - score: {round(score, 4)}')
        count += 1

    time2 = time.time() - time1
    print(f"Total Time Run: {round(time2 / 60,2)}")
    gc.collect()
    print(f'Mean ROC_AUC: {score_mean / FOLDS}')
    del X_tr, X_vl, y_tr, y_vl, clf, score
    return -(score_mean / FOLDS)


xgb_space = {
    # number of estimators
    #'n_estimators': hp.choice('n_estimators', list(range(500, 700, 100))),

    # The maximum depth of a tree, same as GBM.
    # Used to control over-fitting as higher depth will allow model 
    # to learn relations very specific to a particular sample.
    # Should be tuned using CV.
    # Typical values: 3-10
    'max_depth': hp.quniform('max_depth', 7, 23, 1),
    
    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity 
    # (meaning pulling weights to 0). It can be more useful when the objective
    # is logistic regression since you might need help with feature selection.
    'reg_alpha':  hp.uniform('reg_alpha', 0.01, 0.4),
    
    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
    # approach can be more useful in tree-models where zeroing 
    # features might not make much sense.
    'reg_lambda': hp.uniform('reg_lambda', 0.01, .4),
    
    # eta: Analogous to learning rate in GBM
    # Makes the model more robust by shrinking the weights on each step
    # Typical final values to be used: 0.01-0.2
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    
    # colsample_bytree: Similar to max_features in GBM. Denotes the 
    # fraction of columns to be randomly samples for each tree.
    # Typical values: 0.5-1
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, .9),
    
    # A node is split only when the resulting split gives a positive
    # reduction in the loss function. Gamma specifies the 
    # minimum loss reduction required to make a split.
    # Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
    'gamma': hp.uniform('gamma', 0.01, .7),
    
    # more increases accuracy, but may lead to overfitting.
    # num_leaves: the number of leaf nodes to use. Having a large number 
    # of leaves will improve accuracy, but will also lead to overfitting.
    'num_leaves': hp.choice('num_leaves', list(range(20, 250, 10))),
    
    # specifies the minimum samples per leaf node.
    # the minimum number of samples (data) to group into a leaf. 
    # The parameter can greatly assist with overfitting: larger sample
    # sizes per leaf will reduce overfitting (but may lead to under-fitting).
    'min_child_samples': hp.choice('min_child_samples', list(range(100, 250, 10))),
    
    # subsample: represents a fraction of the rows (observations) to be 
    # considered when building each subtree. Tianqi Chen and Carlos Guestrin
    # in their paper A Scalable Tree Boosting System recommend 
    'subsample': hp.choice('subsample', [0.2, 0.4, 0.5, 0.6, 0.7, .8, .9]),
    
    # randomly select a fraction of the features.
    # feature_fraction: controls the subsampling of features used
    # for training (as opposed to subsampling the actual training data in 
    # the case of bagging). Smaller fractions reduce overfitting.
    'feature_fraction': hp.uniform('feature_fraction', 0.4, .8),
    
    # randomly bag or subsample training data.
    'bagging_fraction': hp.uniform('bagging_fraction', 0.4, .9)
    
    # bagging_fraction and bagging_freq: enables bagging (subsampling) 
    # of the training data. Both values need to be set for bagging to be used.
    # The frequency controls how often (iteration) bagging is used. Smaller
    # fractions and frequencies reduce overfitting.
}   


#################################################################################################
def plot_proba_distr(y = None, probabilities = None):

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)

    print(type(y))
    print(type(probabilities))

    #ax.hist(probabilities[y==1], bins=50, label = "Positives (Fraud)", color = 'r')
    ax.hist(probabilities[y==0], bins=50, label = "Negatives (No Fraud)", alpha = 0.5, color = 'skyblue')

    ax.set_xlabel("probability", fontsize = 25)
    ax.set_ylabel("records", fontsize = 25)
    ax.legend(fontsize = 15)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc='lower left', fontsize='small')
    fig.tight_layout()
    plt.show()

    
################################################################################################
def plot_density_distr(predictions = None, scores = None):

    if not predictions:
        print("plot_density_distr - Null predictions!")
        exit(1)

    if not scores:
        print("plot_density_distr - Null scores!")
        exit(1)

    print (predictions[0])
    print (scores[0])
    print(type(predictions))
    print(type(scores))
    X = np.asarray(predictions, dtype=object)
    Y = np.asarray(scores, dtype=object)

    print(X[10:])
    print(Y[10:])
    
    #scatter
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X, Y, color = 'blue', s=2, edgecolors='k')
    #ax.set_xlabel('predictions')
    #ax.set_ylabel('scores')
    #ax.set_xticks(())
    #ax.set_yticks(())
    #ax.legend(loc='lower left', fontsize='small')
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(predictions, scores, color = 'blue', s=2, edgecolors='k')
    ax.set_xlabel('predictions')
    ax.set_ylabel('scores')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc='lower left', fontsize='small')
    fig.tight_layout()
    plt.show()

    

#############################################################################################################
def plot_feature_corellation(df = None):

    fig, ax = plt.subplots(figsize=(15,15))
    # use a ranked correlation to catch nonlinearities
    corr = df[[col for col in train.columns if col != 'TransactionID']].sample(100100).corr(method='spearman')
    _ = sns.heatmap(corr, annot=True,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)

    plt.title('Correlation Matrix')
    plt.show()

#############################################################################################################
def plot_best_feature_cor(df = None, selected_columns = None):
    plt.style.use('ggplot')

    scatter_data = df[selected_columns]
    
    axs = pd.plotting.scatter_matrix( scatter_data,
                                     alpha=0.2,
                                     figsize=(10, 10),
                                     diagonal='kde')
    
    n = len(scatter_data.columns)
    for x in range(n):
        for y in range(n):
            # to get the axis of subplots
            ax = axs[x, y]
            # to make x axis name vertical  
            ax.xaxis.label.set_rotation(90)
            # to make y axis name horizontal 
            ax.yaxis.label.set_rotation(0)
            # to make sure y axis names are outside the plot area
            ax.yaxis.labelpad = 50

    plt.show()  
    
#############################################################################################################
def plot_confusion_matrix(y = None, predictions = None):
    
    cm_train = confusion_matrix(y, predictions)
    print(" Confusion matrix train\n", cm_train)
    #cm_test = confusion_matrix(y_test, y_test_predictions)
    #print ("Confusion matrix test\n", cm_test)
    
    fig = plt.figure()
    plt.clf()
    #ax = fig.add_subplot(111)
    #ax.set_aspect(1)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    annot_kws = {"ha": 'left',"va": 'top'}
    res = sns.heatmap(cm_train,
                      annot=True,
                      annot_kws=annot_kws,
                      vmin=0.0,
                      #vmax=100.0,
                      fmt='.2f', cmap=cmap)
    res.invert_yaxis()
    plt.yticks([0.5,1.5], ["No Fraud", "Fraud"], va='center')
    plt.title('Confusion Matrix')
    plt.show()

#############################################################################################################
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
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='No Skill Clf', alpha=.8)
    
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

    #macro T-P
    """
    all_fpr = np.array( tprs )
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(all_fpr)):
        mean_tpr +=  interp(all_fpr, fprs[i], tprs[i]) 
    mean_tpr /= len(all_fpr)

    macro_fprs = all_fpr
    macro_tprs = mean_tpr
    macro_roc_auc = auc(macro_fprs, macro_tprs)

    ax.plot( macro_fprs, macro_tprs, label = "Macro-avg {0:0.2f}".format(macro_roc_auc), color = 'navy', linestyle = ':', linewidth = 4)
    """
    
    # fprp tune and show the plot.
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    plt.show()
    return (f, ax)
#############################################################################################################
def scatter_targets(X=None, y=None):

    print("Scatter X shape", X.shape)
    print("Scatter X head\n", X.head)
    
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1, 1, 1)
    #ax.scatter(X[y==0,"C1"], X[y==0,"C2"], color='blue', s=2, label='y=0')
    #ax.scatter(X[y!=0,0], X[y!=0,1], color='red', s=2, label='y=1')
    ax.scatter(X[:,0], X[:,1], c = y, cmap=cm_bright, edgecolors='k')
    ax.set_xlabel('X[:,0]')
    ax.set_ylabel('X[:,1]')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend(loc='lower left', fontsize='small')

    fig.tight_layout()
    plt.show()
    #f.savefig('result.png')

    return
#############################################################################################################
def plot_precision_recall(y = None, y_proba = None, precisions = None, recalls = None, labels = None):
    """
    P-R
    """
    fig = plt.figure(figsize=(15,10))
    axis = fig.add_subplot(111)

    #plot iso f lines
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('iso f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    for i in range(len(recalls)):
        if i < len(recalls) - 1:
            axis.step(x = recalls[i], y = precisions[i], label=labels[i], linewidth=1)
        else:
            axis.step(x = recalls[i], y = precisions[i], label=labels[i], linewidth=2, color='black')

    #plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
    axis.set_xlabel('Recall')
    axis.set_ylabel('Precision')
    axis.legend(title = "P-R", loc='lower left', fontsize='small')
    fig.tight_layout()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.show()
    #f.savefig('result.png')
    
    
    return 0
    ###############
    n_classes = 2
    print("Pre-Rec N classes", n_classes)

    print("Pre-Rec y ")
    print(y.shape)
    print(y.head)
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        #prec & recall
        precision[i], recall[i], _ = precision_recall_curve(y[:, i],
                                                            y_score[:, i])
        #avg prec
        average_precision[i] = average_precision_score(y[:, i],
                                                       y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    plt.show()

    
#############################################################################################################
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
    Fast auc eval function for lgbm.
    """
    return 'auc', fast_auc(y_true, y_pred), True

def get_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
#############################################################################################################


def make_predictions_lgbm(X_train = None,
                          y_train = None,
                          X_test = None,
                          nfolds = 5,
                          algo_act = None,
                          verbose = 100,
                          early_stopping_rounds = 10,
                          stratify = True,
                          seed = 123,
                          shuffle = True,
                          eval_metric = "auc",
                          score_averaging = "usual",
                          plot_feature_importance = False,
                          filter_best_features = False,
                          use_preselected_best_features = False,
                          best_variables_list = [],
                          importance_threshold = 0.001
):

    time_beg = time.time()

    #feature selection
    if filter_best_features:
        sfm = SelectFromModel(algo_act, threshold = importance_threshold)

        print("Info: Finding most important features...")
        sfm.fit(X_train, y_train.values.ravel())

        feature_idx = sfm.get_support(indices=True)
        feature_names = X_train.columns.values[feature_idx]
            
        print("Info: SelectFromModel features:")
        print(type(feature_names))
        print(feature_names.tolist())
        print("Info: Selected features =", len(feature_names))
        #this tranforms to numpy array
        #X_train = sfm.transform(X_train) if not X_train.empty else None
        #X_test = sfm.transform(X_test) if not X_test.empty else None

        #keep DFs
        X_train = X_train[feature_names]
        X_test = X_test[feature_names]

    if use_preselected_best_features:
        #keep DFs
        X_train = X_train[best_variables_list]
        X_test = X_test[best_variables_list]

    #columns
    columns = X_train.columns

    if X_train.empty:
        print("Error: Empty training dataset")
        exit(1)
    
    if stratify:
        folds = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)
    else:
        folds = KFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)

    #predictors = X_train.columns.values.tolist()[2:]

    if score_averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X_train), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    elif score_averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X_train), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    # list of scores on folds
    vl_scores = []
    tr_scores = []
    feature_importance = pd.DataFrame()
    accuracy_train_list = []
    accuracy_valid_list = []
    for ifold, (tr_idx, vl_idx) in enumerate(folds.split(X_train, y_train)):

        print("Info: Fold:", ifold)
        print ("Info: Train idx", tr_idx)
        print ("Info: Valid idx", vl_idx)

        # use iloc instead of X_train[tr_idx]
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[vl_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        print("Info: X train size %i, X validation size %i"%(len(X_tr),len(X_vl)))

        #fit
        print('Info: Start fitting estimator')
        gc.collect()
        print('Info: First gc done')
        gc.collect()
        print('Info: Second gc done. Now fitting...')

        algo_act.fit(X_train,
                     y_train, 
                     eval_set=[(X_tr, y_tr), (X_vl, y_vl)],
                     eval_metric=eval_auc if eval_metric == "auc" else None,
                     verbose=verbose,
                     early_stopping_rounds=early_stopping_rounds)

        #predictions
        y_pred_train = algo_act.predict_proba(X_tr)[:, 1]
        y_pred_valid = algo_act.predict_proba(X_vl)[:, 1]
        y_pred_test = algo_act.predict_proba(X_test,
                                             num_iteration=algo_act.best_iteration_)[:, 1]

        y_fold_pred_train = algo_act.predict(X_tr)
        y_fold_pred_valid = algo_act.predict(X_vl)
        
        #AUC
        if score_averaging == 'usual':
            #val score
            x_auc_vl_score = roc_auc_score(y_vl, y_pred_valid)
            print("Info: Fold %i Valid AUC %f"%(ifold, x_auc_vl_score))
            oof[vl_idx] = y_pred_valid.reshape(-1, 1)
            vl_scores.append(x_auc_vl_score)

            #test score
            x_auc_tr_score = roc_auc_score(y_tr, y_pred_train)
            print("Info: Fold %i Train AUC %f"%(ifold, x_auc_tr_score))
            tr_scores.append(x_auc_tr_score)

            #prediction
            prediction += y_pred_test.reshape(-1, 1)

        elif score_averaging == 'rank':
            #val score
            x_auc_score = roc_auc_score(y_vl, y_pred_valid)
            print("Info: Fold %i Valid AUC %f"%(ifold, x_auc_score))
            oof[vl_idx] = y_pred_valid.reshape(-1, 1)
            vl_scores.append(x_auc_score)
                                  
            prediction += pd.Series(y_pred_test).rank().values.reshape(-1, 1)

        #accuracy       
        accuracy_score_train = accuracy_score(y_tr, y_fold_pred_train)
        accuracy_train_list.append(accuracy_score_train)

        accuracy_score_valid =  accuracy_score(y_vl, y_fold_pred_valid)
        accuracy_valid_list.append(accuracy_score_valid)
    
        print("Info: Accuracy (acc score) train = %f valid = %f"%( accuracy_score_train, accuracy_score_valid) )
    
        #importance
        if plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = algo_act.feature_importances_
            fold_importance["fold"] = nfolds + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    #avg pred
    prediction /= nfolds

    #output
    print('\nInfo: Validation Mean AUC: {0:.4f}, std: {1:.4f}.'.format(np.mean(vl_scores),
                                                               np.std(vl_scores)))
    print('Info: Training Mean AUC: {0:.4f}, std: {1:.4f}.'.format(np.mean(vl_scores),
                                                             np.std(vl_scores)))
    #accuracy
    print("Info: Mean valid accuracy", st.mean(accuracy_valid_list))
    print("Info: Mean train accuracy", st.mean(accuracy_train_list))
    
    #importance
    if plot_feature_importance:
        feature_importance["importance"] /= nfolds
        cols = feature_importance[["feature",
                                   "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                        ascending=False)[:25].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        #print 'em
        print("Info: Feature importance:")
        df_best_features = best_features.sort_values(by="importance", ascending=False)
        print(df_best_features.groupby('feature',
                                       as_index = False).mean().sort_values(by=['importance'],
                                                                            ascending = False))
        
        #plot 'em
        plt.figure(figsize=(15, 20))
        sns.barplot(x="importance",
                    y="feature",
                    data=best_features.sort_values(by="importance", ascending=False),
                    color = 'blue')
        plt.title('LGBM Feature Ranking (average over k-folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')
        
    #done
    delta_time = time.time() - time_beg
    print(f"Total training time: {round(delta_time / 60, 2)}")


def make_predictions_skl(X_train = None,
                         y_train = None,
                         X_test = None,
                         nfolds = 2,
                         stratify = False,
                         seed= 123,
                         shuffle = True,
                         algo_act = None,
                         algo_opt = None,
                         optimize = False,
                         filter_best_features = False,
                         use_preselected_best_features = False,
                         best_variables_list = [],
                         importance_threshold = 0.001):
    time_beg = time.time()

    #feature selection
    if filter_best_features:
        sfm = SelectFromModel(algo_act, threshold = importance_threshold)

        print("Finding most important features...")
        sfm.fit(X_train, y_train.values.ravel())

        feature_idx = sfm.get_support(indices=True)
        feature_names = X_train.columns.values[feature_idx]
            
        print("SelectFromModel features:")
        print(type(feature_names))
        print(feature_names.tolist())
        print("Selected features =", len(feature_names))
        
        #Note: this tranforms to numpy array
        #X_train = sfm.transform(X_train) if not X_train.empty else None
        #X_test = sfm.transform(X_test) if not X_test.empty else None

        #keep DFs
        X_train = X_train[feature_names]
        X_test = X_test[feature_names]

    if use_preselected_best_features:
        #keep DFs
        X_train = X_train[best_variables_list]
        X_test = X_test[best_variables_list]
        
    #column names
    columns = X_train.columns.values

    print("Number of features:", len(columns))
    
    #stratify
    if stratify:
        folds = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)
    else:
        folds = KFold(n_splits=nfolds, random_state=seed, shuffle=shuffle)

    y_scores = []
    y_proba = []
    y_valid = []
    labels = []
    precisions = []
    recalls = []
    y_predict_proba_validation_total = np.zeros(X_train.shape[0])  # or np.zeros((len(X_train), 1))
    y_predict_validation_total = []
    y_train_oof = np.zeros(X_train.shape[0])
    list_roc_auc_score = []
    list_tpr = []
    list_fpr = []
    list_auc = []
    feature_importance = pd.DataFrame()
    y_application = np.zeros(X_test.shape[0]) #np.zeros((len(X_test), 1))
    accuracy_train_list = []
    accuracy_valid_list = []
    for ifold, (train_index, validation_index) in enumerate(folds.split(X_train, y_train)):

        print("Fold:", ifold)
        print ("\tTrain X idx", train_index)
        print ("\tTrain y idx", validation_index)

        # use iloc instead of X_train[train_index]
        X_fold_train, X_fold_valid = X_train.iloc[train_index, :], X_train.iloc[validation_index, :]
        y_fold_train, y_fold_valid = y_train.iloc[train_index], y_train.iloc[validation_index]

        print("\tX train size %i, X validation size %i"%(len(X_fold_train), len(X_fold_valid)))
        print("\ty train size %i, y validation size %i"%(len(y_fold_train), len(y_fold_valid)))
        
        #opt on first fold
        if optimize and ifold == 0:
            print("Optimizing hyperparams...")
            param_grid = {'max_depth': range(2,4,2),
                          'min_samples_split': range(2, 4, 2),
                          'n_estimators': range(200, 501, 200)}
    
            scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

            clf = GridSearchCV(estimator = algo_opt,
                               param_grid = param_grid,
                               scoring = scoring,
                               refit='AUC',
                               return_train_score=False)
        
            clf.fit(X_fold_train, y_fold_train.values.ravel())
            print('Best score (AUC): ', clf.best_score_)
            print('Best hyperparameters (max AUC): ', clf.best_params_)
            print('Best parameters set:')
            best_parameters = clf.best_estimator_.get_params()
            for param_name in sorted(param_grid.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]) )

        #fit
        print("Fitting algo...")
        algo_act.fit(X_fold_train, y_fold_train)

        #silly predictions on training
        y_fold_predict_train = algo_act.predict(X_fold_train)
        
        #predictions on validation sample
        y_fold_predict_valid = algo_act.predict(X_fold_valid)
        y_predict_validation_total.append(y_fold_predict_valid.reshape(-1, 1))

        #predictions on application (test) sample
        #y_appl_pred = algo_act.predict_proba(X_test)[:, 1] # on validation fold
        #y_application += pd.Series(y_appl_pred).rank().values.reshape(-1,1) / nfolds
        y_application += algo_act.predict_proba(X_test)[:, 1] / nfolds
        
        #predictions on train sample - for Prec-Rec
        #y_scores_tr = algo_act.predict(X_fold_trainain)#[:, 1]
        #y_scores += y_scores_tr.reshape(-1, 1) # train. Not sure 100%
        #y_score += algo_act.decision_function(X_vl)
        #y_valid += y_vl
        
        #predicted prob
        y_predict_proba_validation = algo_act.predict_proba(X_fold_valid)[:, 1] # on validation fold
        y_train_oof[validation_index] = y_predict_proba_validation # train validation ???

        y_predict_proba_train = algo_act.predict_proba(X_train)[:, 1] # on entire train
        y_predict_proba_validation_total += y_predict_proba_train / nfolds

        ## must be done also on X_test

        #y_predict_proba_validation_total += y_predict_proba_validation #.reshape(-1, 1)
        #y_predict_proba_training = algo_act.predict_proba(X_train)[:, 1] # on train ?? NOT SURE
        #y_predict_proba += y_predict_proba_training.reshape(-1, 1)

        #scores
        y_scores.append( algo_act.score(X_fold_valid, y_fold_valid) )

        #see also metrics.mean_absolute_error
        
        #precision -recall on each fold
        precision, recall, _ = precision_recall_curve(y_fold_valid, y_predict_proba_validation)
        pr_rec_auc = auc(recall, precision)
        
        y_valid.append(y_fold_valid)
        y_proba.append(y_predict_proba_validation.reshape(-1, 1))
        precisions.append(precision)
        recalls.append(recall)
        labels.append( 'Fold %d AUC=%.4f' % (ifold+1, pr_rec_auc ) )
        print(f"\tAUC (from P-R) {pr_rec_auc}")

        #roc auc score train
        roc_auc_scorer_train = make_scorer(roc_auc_score, needs_proba=True)(algo_act, X_fold_train, y_fold_train)
        print(f"\tROC AUC Score (scorer - train) {roc_auc_scorer_train}")
        
        #roc auc score Valid
        roc_auc_scorer_valid = make_scorer(roc_auc_score, needs_proba=True)(algo_act, X_fold_valid, y_fold_valid)
        print(f"\tROC AUC Score (scorer) {roc_auc_scorer_valid}")

        roc_auc_score_valid = roc_auc_score(y_fold_valid, y_predict_proba_validation)
        list_roc_auc_score.append(roc_auc_score_valid)
        print(f"\tROC AUC Score (metrics) {roc_auc_score_valid}")
       
        # Compute ROC curve and AUC: use validation sample 
        fpr, tpr, thresholds = roc_curve(y_fold_valid, y_predict_proba_validation)
        list_tpr.append(tpr)
        list_fpr.append(fpr)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = columns
        fold_importance["importance"] = algo_act.feature_importances_
        fold_importance["fold"] = nfolds + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
        
        #accuracy
        accuracy_train =  algo_act.score(X_fold_train, y_fold_train)
        accuracy_valid =  algo_act.score(X_fold_valid, y_fold_valid)
        print("Accuracy (algo score) train = %f valid = %f", accuracy_train, accuracy_valid)
        accuracy_train_list.append(accuracy_train)
        accuracy_valid_list.append(accuracy_valid)
        
        accuracy_score_train =  accuracy_score(y_fold_train, y_fold_predict_train)
        accuracy_score_valid =  accuracy_score(y_fold_valid, y_fold_predict_valid)
        print("Accuracy (acc score) train = %f valid = %f", accuracy_score_train, accuracy_score_valid)
        
        #release mem
        del X_fold_train, X_fold_valid, y_fold_train, y_fold_valid
        gc.collect()

    ####################
    #time
    ####################
    delta_time = time.time() - time_beg
    print(f"Total training time: {round(delta_time / 60, 2)}")
   
    #######################
    # density distributions
    #######################
    #plot_density_distr(predictions = y_predict_validation_total, scores = y_scores)

    #print( y_predict_proba.shape)
    #print( y_predict_proba.head)

    #######################
    # proba distributions
    #######################
    if do_plots: plot_proba_distr(y = y_valid, probabilities = y_proba)
    
    ####################
    #avg
    ####################
    #avg pred
    #y_predict_proba_acc = y_predict_proba.astype(bool)
    #y_predict_proba /= nfolds # or len( y_predict_proba )
    y_predict_proba_acc = y_predict_proba_validation_total.astype(bool)
    y_predict_proba_avg = y_predict_proba_validation_total

    mean_roc_auc_score  = sum(list_roc_auc_score, 0.0)/len( list_roc_auc_score )

    print(f"Mean AUC {mean_roc_auc_score}")

    # out-of-fold predictions on train data
    mean_roc_auf_oof = roc_auc_score(y_train, y_train_oof)
    print("OOF AUC = {}".format(mean_roc_auf_oof))

    ####################
    #display targets
    ####################    
    #scatter_targets(X=X_train, y=y_train)

        
    ####################
    #precision recall
    ####################    
    y_valid = np.concatenate(y_valid)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_valid, y_proba)
    precisions.append(precision)
    recalls.append(recall)
    labels.append('Overall AUC=%.4f' % (auc(recall, precision)))
        
    if do_plots: plot_precision_recall(y = y_valid, y_proba =  y_proba, precisions = precisions, recalls = recalls, labels = labels)

    ####################
    #accuracy
    ####################
    print("Mean valid accuracy", st.mean(accuracy_valid_list))
    print("Mean train accuracy", st.mean(accuracy_train_list))

    ########################
    #plot ROC curve
    #######################
    if do_plots: plot_roc_curve(fprs = list_fpr, tprs = list_tpr)    

    ##########################
    #plot features importance
    ##########################
    feature_importance["importance"] /= nfolds

    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:25].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)] # type: DataFrame
    df_best_features = best_features.sort_values(by="importance", ascending=False)

    if do_plots: 
        plt.figure(figsize=(16, 12))
        sns.barplot(x="importance",
                    y="feature",
                    data=df_best_features)
        plt.show()

    print("Feature importance:")
    print(df_best_features.groupby('feature',
                                   as_index = False).mean().sort_values(by=['importance'],
                                                                        ascending = False))

    
    ##########################
    #plot features importance
    ##########################
    #if do_plots:
    #    plot_best_feature_cor(df = X_train, selected_columns = cols)

    #######################
    # confusion matrix
    #######################
    #see https://github.com/wcipriano/pretty-print-confusion-matrix/blob/master/confusion_matrix_pretty_print.py

    if do_plots: plot_confusion_matrix(y = y_train, predictions = y_predict_proba_acc)

    #######################
    # feature cor
    #######################    
    #if do_plots:
    #    plot_feature_corellation(df = X_train)
        
    #######################
    # application
    #######################
    if do_sub:
        df_sub['isFraud'] = y_application
        df_sub.to_csv('skl.csv')#, index=False)
        print("Submission:\n", df_sub.head())

        df_sub.loc[ df_sub['isFraud']>0.90, 'isFraud' ] = 1
        pred_frauds = float( df_sub[df_sub['isFraud']==1].sum() )
        print("Predicted %f frauds"%( pred_frauds ) )
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(df_sub['isFraud'], bins = 50, histtype='step', fill=True, alpha = 0.3)
        ax.set_xlabel('Probability',fontsize=18)
        ax.set_ylabel('Frequency',fontsize=18)
        ax.set_title('Probibility Density Distribution')
        fig.tight_layout()
        plt.show()

#=====================================================================
def make_predictions_xgb(X_train = None,
                         y_train = None,
                         X_test = None,
                         n_trees = 100,
                         FOLDS = 5,
                         params = None ):

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
    columns = X_train.columns.values
    y_oof = np.zeros(X_train.shape[0])
    score_mean = 0.
    for ifold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

        if params:
            clf = xgb.XGBClassifier(
                n_estimators = n_trees,
                random_state = 4,
                verbose = True, 
                tree_method = 'gpu_hist', 
                **params
            )
        else:
            clf = xgb.XGBClassifier(
                n_estimators = n_trees,
                random_state = 4,
                verbose = True, 
                tree_method = 'gpu_hist'
            )
            

        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        clf.fit(X_tr, y_tr)

        y_pred_train = clf.predict_proba(X_vl)[:,1]

        y_oof[val_idx] = y_pred_train
        
        score = make_scorer(roc_auc_score, needs_proba=True)(clf, X_vl, y_vl)

        print(f'Fold {ifold} AUC: {round(score, 4)}')
        
        score_mean += score/FOLDS

        del X_tr, X_vl, y_tr, y_vl, score
        gc.collect()

    time2 = time.time() - time1

    
    print("\nMEAN AUC = {}".format(score_mean))
    print("OOF AUC = {}".format(roc_auc_score(y_train, y_oof)))
    #feature_important = clf.get_booster().get_score(importance_type="weight")
    #keys = list(feature_important.keys())
    #values = list(feature_important.values())

    #data = pd.DataFrame(data=values,
    #                    index=keys,
    #                    columns=["score"]).sort_values(by = "score",
    #                                                   ascending=False)

    # Top 10 features
    #data.head(20)

def make_predictions_simple_xgb(X_train = None,
                                y_train = None,
                                X_test = None,
                                FOLDS = 5):

    print("Train shape", X_train.shape)
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma = 0.2,
        alpha = 4,
        missing = -1,
        tree_method='gpu_hist'
    )

    NFOLDS = 5
    kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=123)

    y_preds = np.zeros(X_test.shape[0])
    y_oof = np.zeros(X_train.shape[0])
    score = 0

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    
        X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        clf.fit(X_tr, y_tr)
        y_pred_train = clf.predict_proba(X_vl)[:,1]
        y_oof[val_idx] = y_pred_train
        print("FOLD: ",fold,' AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))
        score += roc_auc_score(y_vl, y_pred_train) / NFOLDS
        y_preds+= clf.predict_proba(X_test)[:,1] / NFOLDS
    
        del X_tr, X_vl, y_tr, y_vl
        gc.collect()
    
    
    print("\nMEAN AUC = {}".format(score))
    print("OOF AUC = {}".format(roc_auc_score(y_train, y_oof)))

    
# prediction ============================================
print("Info: training session...")
if do_make_predictions_skl:
    make_predictions_skl(X_train = X_train,
                         y_train = y_train,
                         X_test = X_test,
                         nfolds = 5,
                         stratify = True,
                         algo_act = clf_skl_act,
                         algo_opt = clf_skl_opt,
                         optimize = skl_optimize,
                         filter_best_features = skl_filter_best_features,
                         use_preselected_best_features = skl_use_preselected_best_features,
                         best_variables_list = skl_best_variables_list,
                         importance_threshold = skl_importance_threshold)

if do_make_predictions_lgbm:
    make_predictions_lgbm(X_train = X_train,
                          y_train = y_train,
                          X_test = X_test,
                          nfolds = 5,
                          algo_act = clf_lgbm_act,
                          verbose = 100,
                          early_stopping_rounds = 100,
                          stratify = True,
                          seed = 123,
                          shuffle = True,
                          eval_metric = "auc",
                          score_averaging = "usual",
                          plot_feature_importance = True,
                          filter_best_features = lgbm_filter_best_features,
                          use_preselected_best_features = lgbm_use_preselected_best_features,
                          best_variables_list = lgbm_best_variables_list,
                          importance_threshold = lgbm_importance_threshold)


if do_make_predictions_xgb:
    # Set algorithm parameters
    if False:
        best = fmin(fn = xgb_objective,
                    space = xgb_space,
                    algo = tpe.suggest,
                    max_evals = 27)

        # Print best parameters
        best_params = space_eval(xgb_space, best)
        best_params['max_depth'] = int(best_params['max_depth'])
        
        
    best_params = {'bagging_fraction': 0.8993155305338455,
                   'colsample_bytree': 0.7463058454739352,
                   'feature_fraction': 0.7989765808988153,
                   'gamma': 0.6665437467229817,
                   'learning_rate': 0.013887824598276186,
                   'max_depth': 16,
                   'min_child_samples': 170,
                   'num_leaves': 220,
                   'reg_alpha': 0.39871702770778467,
                   'reg_lambda': 0.24309304355829786,
                   'subsample': 0.7}

    print("Best XGB params:\n", best_params)
    make_predictions_xgb(X_train = X_train,
                         y_train = y_train,
                         X_test = X_test,
                         n_trees = xgb_n_trees,
                         FOLDS = 5)
                         #params = best_params)


if do_make_predictions_simple_xgb:
    make_predictions_simple_xgb(X_train = X_train,
                                y_train = y_train,
                                X_test = X_test,
                                FOLDS = 5)
    
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



