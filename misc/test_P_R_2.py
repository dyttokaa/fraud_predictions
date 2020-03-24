

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from scipy import interp

X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=10.0,
    random_state=12345)

FOLDS = 5
AUCs = []
AUCs_proba = []

precision_combined = []
recall_combined = []
thresholds_combined = []

#X_ = pred_features.as_matrix()
#Y_ = pred_true.as_matrix()

#k_fold = cross_validation.KFold(n=len(pred_features), n_folds=FOLDS,shuffle=True,random_state=None)
k_fold = KFold(n_splits=FOLDS, shuffle=True, random_state=12345)

#clf = svm.SVC(kernel='linear', C = 1.0)
clf =SVC(kernel='linear', C=1.0, probability=True, random_state=12345)


mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []
reversed_mean_precision = 0.0
mean_recall = np.linspace(0, 1, 100)
all_precision = []

for i, (train_index, test_index) in enumerate(k_fold.split(X)):
    xtrain, xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]

    test_prob = clf.fit(xtrain,ytrain).predict(xtest)
    precision, recall, thresholds = precision_recall_curve(ytest, test_prob, pos_label=2)
    reversed_recall = np.fliplr([recall])[0]
    reversed_precision = np.fliplr([precision])[0]
    reversed_mean_precision += interp(mean_recall, reversed_recall, reversed_precision)
    reversed_mean_precision[0] = 0.0

    AUCs.append(auc(recall, precision))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

reversed_mean_precision /= FOLDS
reversed_mean_precision[0] = 1
mean_auc_pr = auc(mean_recall, reversed_mean_precision)
plt.plot(mean_recall,  np.fliplr([reversed_mean_precision])[0], 'k--',
         label='Mean precision (area = %0.2f)' % mean_auc_pr, lw=2)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall')
plt.legend(loc="lower right")
plt.show()
print ("AUCs: ",  sum(AUCs) / float(len(AUCs)))
