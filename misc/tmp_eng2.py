
    #feature selection
    if filter_best_features:
        sfm = SelectFromModel(algo_act, threshold = 0.001)

        print("Finding most important features...")
        sfm.fit(X_train, y_train.values.ravel())

        feature_idx = sfm.get_support(indices=True)
        feature_names = columns[feature_idx]
            
        print("SelectFromModel features:")
        print(type(feature_names))
        print(feature_names)
        
        X_train = transform(X_train) if X_train else None
        X_test = transform(X_test) if X_test else None

    #column names
    time_beg = time.time()
    columns = X_train.columns.values

    print("Number of features:", len(columns))
    
    #stratify
    print("Folding...")
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
            param_grid = {'max_depth': range(2,4,1),
                          'min_samples_split': range(2, 4, 2),
                          'n_estimators': range(100, 401, 100)}
    
            scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}

            clf = GridSearchCV(estimator = clf_opt,
                               param_grid = param_grid,
                               scoring = scoring,
                               refit='AUC',
                               return_train_score=True)
        
            clf.fit(X_fold_train, y_fold_train.values.ravel())
            print('Best score (AUC): ', clf.best_score_)
            print('Best hyperparameters (max AUC): ', clf.best_params_)
            print('Best parameters set:')
            best_parameters = clf.best_estimator_.get_params()
            for param_name in sorted(param_grid.keys()):
                print('\t%s: %r' % (param_name, best_parameters[param_name]) )

        #find most important features
        """
        if find_features and ifold == 0:
            print("Find most important features...")
            sfm.fit(X_fold_train, y_fold_train.values.ravel())

            feature_idx = sfm.get_support(indices=True)
            feature_names = columns[feature_idx]
            
            print("SelectFromModel features:")
            print(type(feature_names))
            print(feature_names)
            
            #for f_index in feature_idx:
            #    print(columns[f_index])
        """
        #fit
        print("Fitting algo...")
        algo_act.fit(X_fold_train, y_fold_train)

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
        fold_accuracy = accuracy_score(y_fold_valid, y_fold_predict_valid)
        #print(f"\tAccuracy {fold_accuracy}")
        #print(type(y_fold_valid))  pandas Series
        #print(type(y_fold_predict_valid)) numpy array
        
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
    #accuracy1 = max(mean(y_valid.values), 1. - mean(y_valid.values))
    #print(f"Null Accuracy {accuracy1}")

    print("UNFINISHED", type(y_valid)) 
    """
    print("TMP2",  y_valid.shape)
    print("TMP2\n",  y_valid.head)

    print("TMP2",  y_predict_validation_total.shape)
    print("TMP2\n",  y_predict_validation_total.head)

    accuracy2 = accuracy_score(y_valid, y_predict_validation_total)
    print(f"Score Accuracy {accuracy2}")
    """

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

    if do_plots: 
        plt.figure(figsize=(16, 12))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
        plt.show()

    print("Feature importance:")
    print(best_features.sort_values(by="importance", ascending=False))
    
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

    
# prediction ============================================

if do_make_predictions_skl:
    make_predictions_skl(X_train = X_train,
                         y_train = y_train,
                         X_test = X_test,
                         nfolds = 5,
                         stratify = True,
                         algo_act = clf_skl_act,
                         algo_opt = clf_skl_opt,
                         optimize = False,
                         filter_best_features = False,
                         importance_threshold = 0.001)

