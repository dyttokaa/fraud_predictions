# https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt

import xtools as xt
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

#import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


def CalcOutliers(df_num): 
    """
    https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
    """
    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # seting the cut line to both higher and lower values
    # You can change this value
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) # printing total number of values in lower cut of outliers
    print('Identified upper outliers: %d' % len(outliers_higher)) # printing total number of values in higher cut of outliers
    print('Total outlier observations: %d' % len(outliers_total)) # printing total number of values outliers of both sides
    print('Non-outlier observations: %d' % len(outliers_removed)) # printing total number of non outlier values
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4)) # Percentual of outliers in points
    
    return

#plot distr
# https://www.kaggle.com/kabure/baseline-fraud-detection-eda-interactive-views
def plot_distribution(df, var_select=None, title=None, bins=1.0): 
    # Calculate the correlation coefficient between the new variable and the target
    tmp_fraud = df[df['isFraud'] == 1]
    tmp_no_fraud = df[df['isFraud'] == 0]    
    corr = df['isFraud'].corr(df[var_select])
    corr = np.round(corr,3)
    tmp1 = tmp_fraud[var_select].dropna()
    tmp2 = tmp_no_fraud[var_select].dropna()
    hist_data = [tmp1, tmp2]
    
    group_labels = ['Fraud', 'No Fraud']
    colors = ['seagreen','indianred', ]

    fig = ff.create_distplot(hist_data,
                             group_labels,
                             colors = colors, 
                             show_hist = True,
                             curve_type='kde', 
                             bin_size = bins
                            )

    fig['layout'].update(title = title+' '+'(corr target ='+ str(corr)+')')

    fig.show()

#missing data
def missing_data(df = None):
    missing_values_count = df.isnull().sum()
    print (missing_values_count[0:50])
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()
    print ("% of missing data = ",(total_missing/total_cells) * 100)

#resume
def resume(df = None):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

#imbalance
def show_imbalance(df = None):
    x = df['isFraud'].value_counts().index #The index (axis labels) of the Series.
    y = df['isFraud'].value_counts().values #Series as ndarray or ndarray-like depending on the dtype

    trace2 = go.Bar(
        x=x ,
        y=y,
        marker=dict(
            color=y,
            colorscale = 'Viridis',
            reversescale = True
        ),
        name="Imbalance",    
    )
    layout = dict(
        title="Data imbalance - isFraud",
        #width = 900, height = 500,
        xaxis=go.layout.XAxis(
            automargin=True),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            #         domain=[0, 0.85],
        ), 
    )
    fig1 = go.Figure(data=[trace2], layout=layout)
    fig1.show()

def show_imbalance_and_stats(df = None):
    print("Transactions % Fraud:")
    print( round( df[['isFraud', 'TransactionDT']]['isFraud'].value_counts(normalize=True) * 100, 2 ) )

    trace0 = go.Bar(
        x=df[['isFraud', 'TransactionDT']].groupby('isFraud')['TransactionDT'].count().index,
        y=df[['isFraud', 'TransactionDT']].groupby('isFraud')['TransactionDT'].count().values,
        marker=dict(
            color=['indianred', 'seagreen']),
    )

    data = [trace0] 
    layout = go.Layout(
        title='Fraud (Target) Distribution <br>## 0: No Fraud | 1: Is Fraud ##', 
        xaxis=dict(
            title='Transaction is fraud', 
            type='category'),
        yaxis=dict(
            title='Count')
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

#target distribution
def show_target_distribution(df = None):
    df['TransactionAmt'] = df['TransactionAmt'].astype(float)
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    plt.figure(figsize=(16,6))

    plt.subplot(121)
    g = sns.countplot(x='isFraud', data=df, )
    g.set_title("Fraud Transactions Distribution \n# 0: No Fraud | 1: Fraud #", fontsize=22)
    g.set_xlabel("Is fraud?", fontsize=18)
    g.set_ylabel('Count', fontsize=18)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
               height + 3,
               '{:1.2f}%'.format(height/total*100),
               ha="center", fontsize=15) 

    perc_amt = (df.groupby(['isFraud'])['TransactionAmt'].sum())
    perc_amt = perc_amt.reset_index()

    plt.subplot(122)
    g1 = sns.barplot(x='isFraud', y='TransactionAmt',  dodge=True, data=perc_amt)
    g1.set_title("% Total Amount in Transaction Amt \n# 0: No Fraud | 1: Fraud #", fontsize=22)
    g1.set_xlabel("Is fraud?", fontsize=18)
    g1.set_ylabel('Total Transaction Amount Scalar', fontsize=18)
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 4,
                '{:1.2f}%'.format(height/total_amt * 100),
                ha="center", fontsize=15) 
    
    plt.show()

#transaction amount quantiles
def trans_amount_quantiles(df = None):
    df['TransactionAmt'] = df['TransactionAmt'].astype(float)
    print("Transaction Amounts Quantiles:")
    print(df['TransactionAmt'].quantile([.01, .025, .1, .25, .5, .75, .9, .975, .99]))

#transaction amoutn values
def trans_amount_values(df = None):
    ##
    plt.figure(figsize=(10, 12))
    plt.suptitle('Transaction Values Distribution', fontsize=22)

    plt.subplot(2,1,1)
    g = sns.distplot(df[df['TransactionAmt'] <= 1000]['TransactionAmt'])
    g.set_title("Transaction Amount Distribution <= 1000", fontsize=18)
    g.set_xlabel("index")
    g.set_ylabel("Probability", fontsize=15)


    plt.subplot(2,1,2)
    g1 = sns.distplot(np.log( df[ df['TransactionAmt'] > 0]['TransactionAmt'] ))
    g1.set_title("All Transaction Amounts (Log) Distribution", fontsize=18)
    g1.set_xlabel("index")
    g1.set_ylabel("Probability", fontsize=15)

    #
    fig, ax = plt.subplots(1, 2, figsize=(18,4))

    time_val = df.loc[df['isFraud'] == 1]['TransactionAmt'].values
    sns.distplot(np.log(time_val), ax=ax[0], color='r')
    ax[0].set_title('Distribution of LOG TransactionAmt, isFraud=1', fontsize=14)
    ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

    time_val = df.loc[df['isFraud'] == 0]['TransactionAmt'].values

    sns.distplot(np.log(time_val), ax=ax[1], color='b')
    ax[1].set_title('Distribution of LOG TransactionAmt, isFraud=0', fontsize=14)
    ax[1].set_xlim([min(np.log(time_val)), max(np.log(time_val))])

    plt.show()
    
    ##
    plt.figure(figsize=(16,12))

    plt.subplot(212)
    g4 = plt.scatter(range(df[df['isFraud'] == 0].shape[0]),
                     np.sort(df[df['isFraud'] == 0]['TransactionAmt'].values), 
                     label='NoFraud', alpha=.2)

    g4 = plt.scatter(range(df[df['isFraud'] == 1].shape[0]),
                     np.sort(df[df['isFraud'] == 1]['TransactionAmt'].values), 
                     label='Fraud', alpha=.2, color = 'orange')

    g4= plt.title("ECDF \nFRAUD and NO FRAUD Transaction Amount Distribution", fontsize=18)
    g4 = plt.xlabel("Index")
    g4 = plt.ylabel("Amount Distribution", fontsize=15)
    g4 = plt.legend()

    ##
    #plt.figure(figsize=(16,12))

    plt.subplot(321)
    g = plt.scatter(range(df[df['isFraud'] == 1].shape[0]), 
                    np.sort(df[df['isFraud'] == 1]['TransactionAmt'].values), 
                    label='isFraud', alpha=.4, color = "orange")
    plt.title("FRAUD - Transaction Amount ECDF", fontsize=18)
    plt.xlabel("Index")
    plt.ylabel("Amount Distribution", fontsize=12)

    plt.subplot(322)
    g1 = plt.scatter(range(df[df['isFraud'] == 0].shape[0]),
                     np.sort(df[df['isFraud'] == 0]['TransactionAmt'].values), 
                     label='NoFraud', alpha=.2)
    g1= plt.title("NO FRAUD - Transaction Amount ECDF", fontsize=18)
    g1 = plt.xlabel("Index")
    g1 = plt.ylabel("Amount Distribution", fontsize=15)

    plt.suptitle('Individual ECDF Distribution', fontsize=22)

    plt.show()


    #
    tmp = df[['TransactionAmt', 'isFraud']][0:100000]
    plot_distribution(tmp[(tmp['TransactionAmt'] <= 800)], 'TransactionAmt', 'Transaction Amount Distribution', bins=10.0,)
    
#transaction quantiles
def trans_quantiles(df = None):
    q = [.01, .1, .25, .3, .4, .5, .6, .7, .75, .8, .85, .9, .95, .99]
    df_q = pd.concat([df[df['isFraud'] == 1]['TransactionAmt']
                      .quantile(q)
                      .reset_index(), 
                      df[df['isFraud'] == 0]['TransactionAmt']
                      .quantile(q)
                      .reset_index()],
                     axis=1,
                     keys=['Fraud', "No Fraud"]) 

    print (df_q)
    plt.figure(figsize=(8,6))
    g = plt.scatter( q,
                     np.sort(df[df['isFraud'] == 1]['TransactionAmt'].quantile(q)),
                     label='Fraud', alpha=.6, color = "orange")
    g = plt.scatter( q,
                     np.sort(df[df['isFraud'] == 0]['TransactionAmt'].quantile(q)),
                     label='No Fraud', alpha=.6, color = "blue")

    g = plt.title("AMT transaction Quantiles", fontsize=18)
    g = plt.xlabel("Quantile")
    g = plt.ylabel("Amount Distribution", fontsize=15)
    g = plt.legend()


    plt.show()

#transaction outliers
#considering outlier values that are highest than 3 times the std from the mean
def trans_outliers(df = None, variable = ""):

    df_col = df[variable]
    
    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_col), np.std(df_col)

    # seting the cut line to both higher and lower values
    cut = data_std * 3

    #Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_col if x < lower]
    outliers_higher = [x for x in df_col if x > upper]
    outliers_total = [x for x in df_col if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_col if x > lower and x < upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower))
    print('Identified upper outliers: %d' % len(outliers_higher))
    print('Total outlier observations: %d' % len(outliers_total))
    print('Non-outlier observations: %d' % len(outliers_removed))
    print("Total percentual of Outliers: ", round((len(outliers_total) / len(outliers_removed) )*100, 4))
    
#Distribution Products / Distribution of Frauds by Product
def product_cd_distributions(df = None):
    #
    ctab = pd.crosstab(index = df['ProductCD'],
                     columns = df['isFraud'],
                     normalize='index') * 100
    ctab = ctab.reset_index()
    ctab.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    #
    plt.figure(figsize=(14,10))
    plt.suptitle('ProductCD Distributions', fontsize=22)
    total = len(df)

    #
    #print (df.groupby('ProductCD')['TransactionID'].nunique() )
     
    plt.subplot(221)
    #sns.set(style="darkgrid")
    g = sns.countplot(x='ProductCD',
                      data=df)
    
    g.set_title("ProductCD Distribution", fontsize=19)
    g.set_xlabel("ProductCD Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    ymax = max( [h.get_height() for h in g.patches] )
    g.set_ylim(0,  ymax*1.1)

    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x() + p.get_width()/2.,
               height + 3,
               '{:1.2f}%'.format(height/total*100),
               ha="center",
               fontsize=14) 

    #
    plt.subplot(222)
    g1 = sns.countplot(x='ProductCD',
                       hue='isFraud',
                       data=df)
    plt.legend(title='Fraud',
               loc='best',
               labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x='ProductCD',
                       y='Fraud',
                       data=ctab,
                       color='black',
                       order=['W', 'H',"C", "S", "R"],
                       legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title("Product CD by Target(isFraud)", fontsize=19)
    g1.set_xlabel("ProductCD Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)


    #
    plt.subplot(212)
    maxTrans = 2000
    g3 = sns.barplot(x='ProductCD',
                       y='TransactionAmt',
                       hue='isFraud', 
                       data=df[df['TransactionAmt'] <= maxTrans] )
    g3.set_title("Transaction Amount Distribuition by ProductCD and Target (<%f)"%(maxTrans), fontsize=20)
    g3.set_xlabel("ProductCD Name", fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)

    plt.subplots_adjust(hspace = 0.6, top = 0.85)

    plt.show()

#card quantiles
def card_quantiles(df = None):
    q = [0.01, .025, .1, .25, .3, .4, .5, .6, .7, .75, .8, .9, .975, .99]

    print("Card Features Quantiles: ")
    print(df[['card1', 'card2', 'card3', 'card5']].quantile(q))

    plt.figure(figsize=(8,6))
    g = plt.scatter( q,
                     np.sort(df['card1'].quantile(q)),
                     label='card1',
                     alpha=.4,
                     color = "orange")

    g = plt.scatter( q,
                     np.sort(df['card2'].quantile(q)),
                     label='card2',
                     alpha=.4,
                     color = "red")

    g = plt.scatter( q,
                     np.sort(df['card3'].quantile(q)),
                     label='card3',
                     alpha=.4,
                     color = "blue")

    g = plt.scatter( q,
                     np.sort(df['card5'].quantile(q)),
                     label='card5',
                     alpha=.4,
                     color = "green")
    
    g = plt.title("Card transaction Quantiles", fontsize=18)
    g = plt.xlabel("Card")
    g = plt.ylabel("Quantile", fontsize=15)
    g = plt.legend()


    plt.show()

#card distributions
def card_distributions(df = None):
    #for cards values with low frequencies we set values to "Others
    df.loc[df.card3.isin( df.card3.value_counts()[df.card3.value_counts() < 200].index ), 'card3'] = "Others"
    df.loc[df.card5.isin( df.card5.value_counts()[df.card5.value_counts() < 300].index ), 'card5'] = "Others"

    #cross tables
    ct3 = pd.crosstab(df['card3'], df['isFraud'], normalize='index') * 100
    ct3 = ct3.reset_index()
    ct3.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    ct5 = pd.crosstab(df['card5'], df['isFraud'], normalize='index') * 100
    ct5 = ct5.reset_index()
    ct5.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    #tot
    total = len(df)

    #
    plt.figure(figsize=(10, 20))

    
    plt.subplot(411)
    g = sns.distplot(df[df['isFraud'] == 1]['card1'], label='Fraud', color = "orange")
    g = sns.distplot(df[df['isFraud'] == 0]['card1'], label='NoFraud', color = "dodgerblue")
    g.legend()
    g.set_title("Card 1 Values Distribution by Target", fontsize=16)
    g.set_xlabel("Card 1 Values", fontsize=12)
    g.set_ylabel("Probability", fontsize=12)

    
    plt.subplot(412)
    g1 = sns.distplot(df[df['isFraud'] == 1]['card2'].dropna(), label='Fraud', color = "orange")
    g1 = sns.distplot(df[df['isFraud'] == 0]['card2'].dropna(), label='NoFraud', color = "dodgerblue")
    g1.legend()
    g1.set_title("Card 2 Values Distribution by Target", fontsize=16)
    g1.set_xlabel("Card 2 Values", fontsize=12)
    g1.set_ylabel("Probability", fontsize=12)


    plt.subplot(413)
    g2 = sns.countplot(x='card3', data=df, order=list(ct3.card3.values))
    g22 = g2.twinx()
    gg2 = sns.pointplot(x='card3',
                        y='Fraud',
                        data=ct3, 
                        color='gray',
                        order=list(ct3.card3.values))
    gg2.set_ylabel("% of Fraud Transactions", fontsize=12)
    g2.set_title("Card 3 Values Distribution and % of Transaction Frauds", fontsize=16)
    g2.set_xlabel("Card 3 Values", fontsize=16)
    g2.set_ylabel("Count", fontsize=16)
    for p in g2.patches:
        height = p.get_height()
        g2.text(p.get_x()+p.get_width()/2.,
                height + 25,
                '{:1.2f}%'.format(height/total*100),
                ha="center",
                fontsize = 11) 
    ymax2 = max( [h.get_height() for h in g2.patches] )
    g2.set_ylim(0,  ymax2*1.1)


    plt.subplot(414)
    g3 = sns.countplot(x='card5', data=df, order=list(ct5.card5.values))
    g3t = g3.twinx()
    g3t = sns.pointplot(x='card5',
                        y='Fraud',
                        data=ct5, 
                        color='gray',
                        order=list(ct5.card5.values))
    g3t.set_ylabel("% of Fraud Transactions", fontsize=12)
    g3.set_title("Card 5 Values Distribution and % of Transaction Frauds", fontsize=16)
    g3.set_xticklabels(g3.get_xticklabels(), rotation=90)
    g3.set_xlabel("Card 5 Values", fontsize=16)
    g3.set_ylabel("Count", fontsize=16)
    for p in g3.patches:
        height = p.get_height()
        g3.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",
                fontsize=11) 
    ymax3 = max( [h.get_height() for h in g3.patches] )
    g3.set_ylim(0,  ymax3*1.1)

        
    plt.subplots_adjust(hspace = 0.6, top = 0.95)
    plt.show()        

#card distributions categorical
def card_categorical(df = None, cardNum = 0, force_name = False):
    cardName = 'card%i'%(cardNum)

        
    #cross table
    ct = pd.crosstab(df[cardName], df['isFraud'], normalize='index') * 100
    ct = ct.reset_index()
    ct.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    if force_name:
        if cardNum == 4:
            card_order = ["Discover", "Mastercard", "Visa", "AmExpress"]
        elif cardNum == 6:
            card_order = ["Charge", "Credit", "Debit", "Credit or Debit"]
        else:
            card_order = list(ct[cardName].values)
    else:
        card_order = list(ct[cardName].values)
    #tot
    total = len(df)

    plt.figure(figsize=(14,10))
    plt.suptitle('Card %i Distributions'%(cardNum), fontsize=16)

    ##############################################################
    plt.subplot(221)
    g = sns.countplot(x=cardName, data=df)
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])
    g.set_title("%s Distribution"%(cardName), fontsize=19)
    ymax = max( [h.get_height() for h in g.patches] )
    g.set_ylim(0, ymax*1.1)
    g.set_xlabel("%s Category Names"%(cardName), fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
               height + 3,
               '{:1.2f}%'.format(height/total*100),
               ha="center",fontsize=14) 
    

    plt.subplot(222)
    g1 = sns.countplot(x=cardName,
                       hue='isFraud',
                       data=df,
                       order=card_order
    )
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    g1t = g1.twinx()
    g1t = sns.pointplot(x=cardName,
                       y='Fraud',
                       data=ct, 
                       color='gray',
                       legend=False, 
                       order=card_order
                       )
    #ymaxt = max( [h.get_height() for h in gt.patches] )
    #gt.set_ylim(0, ymaxt*1.1)                 
    g1t.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title("%s by Target(isFraud)"%(cardName), fontsize=19)
    g1.set_xlabel("%s Category Names"%(cardName), fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    
    plt.subplot(212)
    g3 = sns.barplot(x=cardName,
                     y='TransactionAmt',
                     hue='isFraud', 
                     data=df[df['TransactionAmt'] <= 2000] )
    g3.set_title("%s Distribuition by ProductCD and Target"%(cardName), fontsize=18)
    g3.set_xlabel("%s Category Names"%(cardName), fontsize=16)
    g3.set_ylabel("Transaction Values", fontsize=16)

            
    
    plt.subplots_adjust(hspace = 0.6, top = 0.85)   
    plt.show()

#explore the M features
def M_features(df = None, col = '', lim=2000, logy = False):
    if not col:
        print ("Warning: Select column first.")
        return
    #for col in ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
    df[col] = df[col].fillna("Miss")

    #tot
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    
    #cross-tables
    ct = pd.crosstab( index = df[col],
                      columns = df['isFraud'],
                      normalize='index') * 100
    ct = ct.reset_index()
    ct.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    #
    plt.figure(figsize=(20,5))
    plt.suptitle(f'{col} Distributions ', fontsize=18)

    #
    plt.subplot(121)
    g = sns.countplot(x=col,
                      data=df,
                      color = 'blue',
                      order=list(ct[col].values))
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])
    g.set_title(f"{col} Distribution\nCounts and percentage fraud rate", fontsize=16)
    ymax = max( [h.get_height() for h in g.patches] )


    if logy:
        g.set_ylim(0, ymax*10)
        g.set_yscale('log')
    else:
        g.set_ylim(0, ymax*1.1)        
        
    gt = g.twinx()
    gt = sns.pointplot(x=col,
                       y='Fraud',
                       data=ct,
                       order=list(ct[col].values),
                       color='gray',
                       legend=False )
    if gt.patches:
        ymaxt = max( [h.get_height() for h in gt.patches] )
        gt.set_ylim(0, ymaxt)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Counts", fontsize=17)
    for p in gt.patches:
        height = p.get_height()
        gt.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center",fontsize=14) 
        
    perc_amt = (df.groupby(['isFraud', col])['TransactionAmt'].sum() / total_amt * 100).unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    
    plt.subplot(122)
    g1 = sns.boxplot(x=col,
                     y='TransactionAmt',
                     hue='isFraud',
                     data=df[df['TransactionAmt'] <= lim],
                     order=list(ct[col].values))
    
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col, y='Fraud',
                        data=perc_amt,
                        order=list(ct[col].values),
                        color='gray',
                        legend=False, )
    g1t.set_ylim(0,5)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g1.set_title(f"{col} by Transactions dist", fontsize=18)
    g1.set_xlabel(f"{col} Categories", fontsize=16)
    g1.set_ylabel("Transaction Amount(U$)", fontsize=16)
        
    plt.subplots_adjust(hspace=.4, wspace = 0.35, top = 0.80)
    
    plt.show()

#address quantiles
def address_quantiles(df = None):
    Q = [0.01, .025, .1, .25, .5, .75, .90, .975, .99]
    
    print("Card Features Quantiles: ")
    dfQ = df[['addr1', 'addr2']].quantile(Q)


    print (dfQ)

    df_q = pd.concat([df[df['isFraud'] == 1]['addr1']
                      .quantile(Q)
                      .reset_index(), 
                      df[df['isFraud'] == 0]['addr1']
                      .quantile(Q)
                      .reset_index()],
                     axis=1,
                     keys=['Fraud', "No Fraud"]) 

    print("Card Features Quantiles Fraud/No Fraud: ")
    print (df_q)


    #
    plt.figure(figsize=(14,10))
    plt.suptitle('Address Quantiles' , fontsize=16)

    plt.subplot(221)
    g1 = plt.title("Addr 1", fontsize=14)
    g1 = plt.scatter( Q,
                     np.sort(df[df['isFraud'] == 1]['addr1'].quantile(Q)),
                     label='Fraud', alpha=.6, color = "orange")
    g1 = plt.scatter( Q,
                     np.sort(df[df['isFraud'] == 0]['addr1'].quantile(Q)),
                     label='No Fraud', alpha=.4, color = "blue")
    g1 = plt.xlabel("Index")
    g1 = plt.ylabel("A.U.", fontsize=15)
    g1 = plt.legend()

    #
    plt.subplot(222)
    g2 = plt.title("Addr 2", fontsize=14)
    g2 = plt.scatter( Q,
                     np.sort(df[df['isFraud'] == 1]['addr2'].quantile(Q)),
                     label='Fraud', alpha=.6, color = "orange")
    g2 = plt.scatter( Q,
                     np.sort(df[df['isFraud'] == 0]['addr2'].quantile(Q)),
                     label='No Fraud', alpha=.4, color = "blue")
    g2 = plt.xlabel("Index")
    g2 = plt.ylabel("A.U.", fontsize=15)
    g2 = plt.legend()


    #
    plt.subplot(212)
    g3 = plt.title("Addr 1 & 2", fontsize=14)
    g3 = plt.scatter( dfQ.index,
                      np.sort(dfQ['addr1']),
                      label='addr1', alpha=.6, color = "green")
    g3 = plt.scatter( dfQ.index,
                      np.sort(dfQ['addr2']),
                      label='addr2', alpha=.6, color = "red")
    g3 = plt.xlabel("Index")
    g3 = plt.ylabel("A.U.", fontsize=15)
    g3 = plt.legend()
    

    
    plt.subplots_adjust(hspace = 0.6, top = 0.85)
    plt.show()

#col distributions
def col_distr(df = None, col = '', maxval = -1):
    #all values in addr that has less than "maxval" entries to "Others"
    if maxval > 0:
        df.loc[df[col].isin(df[col].value_counts()[df[col].value_counts() <= maxval].index), col] = "Others"


    print("head",df[col].head())
    #cross-table
    ct = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    ct = ct.reset_index()
    ct.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    #tot
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    #
    plt.figure(figsize=(16,14))    
    plt.suptitle(f'{col} Distributions ', fontsize=24)

    print("Column values", ct[col].values)
    #---------------------------------------------------------
    plt.subplot(211)
    g = sns.countplot( x=col, data=df, order=list(ct[col].values))
    gt = g.twinx()
    gt = sns.pointplot(x=col,
                       y='Fraud',
                       data=ct,
                       order=list(ct[col].values),
                       color='gray',
                       legend=False, )
    gt.set_ylim(0, ct['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} Category Names", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
               '{:1.2f}%'.format(height/total*100),
               ha="center",
               fontsize=12) 

    ymax = max( [h.get_height() for h in g.patches] )
    g.set_ylim(0, ymax*1.15)

    #---------------------------------------------------------
    perc_amt = df.groupby(['isFraud', col])['TransactionAmt'].sum() / df.groupby([col])['TransactionAmt'].sum() * 100
    perc_amt = perc_amt.unstack('isFraud')
    perc_amt = perc_amt.reset_index()
    perc_amt.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)
    amt = df.groupby([col])['TransactionAmt'].sum().reset_index()
    perc_amt = perc_amt.fillna(0)

    plt.subplot(212)
    g1 = sns.barplot(x=col,
                     y='TransactionAmt', 
                     data=amt, 
                     order=list(ct[col].values))
    g1t = g1.twinx()
    g1t = sns.pointplot(x=col,
                        y='Fraud',
                        data=perc_amt, 
                        order=list(ct[col].values),
                        color='gray',
                        legend=False, )
    g1t.set_ylim(0, perc_amt['Fraud'].max()*1.1)
    g1t.set_ylabel("%Fraud Total Amount", fontsize=16)
    g1t.set_xticklabels(g.get_xticklabels(),rotation=45)
    g1.set_title(f"{col} by Transactions Total + %of total and %Fraud Transactions", fontsize=20)
    g1.set_xlabel(f"{col} Category Names", fontsize=16)
    g1.set_ylabel("Transaction Total Amount(U$)", fontsize=16)
    g1.set_xticklabels(g.get_xticklabels(),rotation=45)    
    
    for p in g1.patches:
        height = p.get_height()
        g1.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total_amt*100),
                ha="center",fontsize=12) 
        
    #
    plt.subplots_adjust(hspace=.5, top = 0.9)
    plt.show()

#pemailmain distributions
def P_emaildomain_distributions(df = None, N_others_max = -1):

    print("P email examples", df['P_emaildomain'].head(20))
    #group all e-mail domains by the respective enterprises.
    #set as "Others" all values with less than N_others_max entries.
    df.loc[df['P_emaildomain'].isin(['gmail.com', 'gmail']), 'P_emaildomain2'] = 'Google'

    df.loc[df['P_emaildomain'].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                                     'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                                     'yahoo.es']), 'P_emaildomain2'] = 'Yahoo Mail'
    
    df.loc[df['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                     'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                     'outlook.es', 'live.com', 'live.fr',
                                     'hotmail.fr']), 'P_emaildomain2'] = 'Microsoft'
    
    df.loc[df.P_emaildomain.isin(df.P_emaildomain.value_counts()[df.P_emaildomain.value_counts() <= N_others_max ].index),
           'P_emaildomain2'] = "Others"

    df.P_emaildomain2.fillna("NoInf", inplace=True)

    print("Calling col_distr for P email")
    col_distr(df = df, col = 'P_emaildomain2', maxval = N_others_max)

#R emailmain distributions
def Emaildomain_distributions(etype="", df = None, N_others_max = -1):
    
    #group all e-mail domains by the respective enterprises.
    #set as "Others" all values with less than N_others_max entries.

    field = etype+'_emaildomain'
    
    df.loc[df[field].isin(['gmail.com',
                           'gmail']), field] = 'Google'

    df.loc[df[field].isin(['yahoo.com', 'yahoo.com.mx',  'yahoo.co.uk',
                           'yahoo.co.jp', 'yahoo.de', 'yahoo.fr',
                           'yahoo.es']), field] = 'Yahoo Mail'

    df.loc[df[field].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 
                                     'hotmail.es','hotmail.co.uk', 'hotmail.de',
                                     'outlook.es', 'live.com', 'live.fr',
                                     'hotmail.fr']), field] = 'Microsoft'
    
    df.loc[df[field].isin(df[field].value_counts()[df[field].value_counts() <= N_others_max ].index), field] = "Others"

    df[field].fillna("NoInf", inplace=True)


    col_distr(df = df, col = field, maxval = N_others_max)

def P_emaildomain_plot(df = None, pre = ""):
    fig, ax = plt.subplots(1, 3, figsize=(32,10))

    sns.countplot(y=pre+"_emaildomain", ax=ax[0], data = df)
    ax[0].set_title(pre+'_emaildomain', fontsize=14)
    sns.countplot(y=pre+"_emaildomain", ax=ax[1], data = df.loc[data['isFraud'] == 1])
    ax[1].set_title(pre+'_emaildomain isFraud = 1', fontsize=14)
    sns.countplot(y=pre+"_emaildomain", ax=ax[2], data = df.loc[data['isFraud'] == 0])
    ax[2].set_title(pre+'_emaildomain isFraud = 0', fontsize=14)
    plt.show()

def c_features_resume(df):
    lc = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
    print( resume( df[lc] ))
    print(df[lc].describe())

# C features
def c_features_distr(df = None, cf = '', N_others_max = -1):
    col_distr(df = df, col = cf, maxval = N_others_max)

def card_features_distr(df = None, cf = '', N_others_max = -1):
    col = cf
    maxval = N_others_max
    if maxval > 0:
        df.loc[df[col].isin(df[col].value_counts()[df[col].value_counts() <= maxval].index), col] = "Others"


    print("head",df[col].head())
    #cross-table
    ct = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    ct = ct.reset_index()
    ct.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    #tot
    total = len(df)
    total_amt = df.groupby(['isFraud'])['TransactionAmt'].sum().sum()
    #
    plt.figure(figsize=(16,14))    
    #plt.suptitle(f'{col} Distributions ', fontsize=24)

    print("Column values", ct[col].values)

    g = sns.countplot( x=col, data=df, order=list(ct[col].values), color = 'blue')
    gt = g.twinx()
    gt = sns.pointplot(x=col,
                       y='Fraud',
                       data=ct,
                       order=list(ct[col].values),
                       color='gray',
                       legend=False, )
    gt.set_ylim(0, ct['Fraud'].max()*1.1)
    gt.set_ylabel("%Fraud Transactions", fontsize=16)
    g.set_title(f"Most Frequent {col} values and % Fraud Transactions", fontsize=20)
    g.set_xlabel(f"{col} categories", fontsize=16)
    g.set_ylabel("Count", fontsize=17)
    g.set_xticklabels(g.get_xticklabels(),
                      fontweight='light',
                      fontsize='small',
                      rotation=45)

    plt.show()
    

def adjust_time(df = None):
    import datetime
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    df["Date"] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

    df['_Weekdays'] = df['Date'].dt.dayofweek
    df['_Hours'] = df['Date'].dt.hour
    df['_Days'] = df['Date'].dt.day

    return df

#time
def time_delta(df = None):

    # solution to Timedelta column
    # We will use the first date as 2017-12-01 and use the delta time to compute datetime features
    # https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400#latest-579480

    df = adjust_time(df)

    print(f"First Date is {df['Date'].dt.date.min()} and the last date is {df['Date'].dt.date.max()}")
    print(f"Total Difference in days are {(df['Date'].max() - df['Date'].min()).days} Days")

    #Top Days with highest Total Transaction Amount
    col_distr(df = df, col = "_Days", maxval = -1)
    col_distr(df = df, col = "_Weekdays", maxval = -1)
    col_distr(df = df, col = "_Hours", maxval = -1)

def trans_per_day(df = None):
    # Calling the function to transform the date column in datetime pandas object

    #seting some static color options
    color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 
                '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 
                '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']


    df = adjust_time(df)

    dates_temp = df.groupby(df.Date.dt.date)['TransactionAmt'].count().reset_index()

    
        
    # creating the first trace with the necessary parameters
    trace = go.Scatter(x=dates_temp['Date'],
                       y=dates_temp.TransactionAmt,
                       opacity = 0.8,
                       line = dict(color = color_op[7]),
                       name= 'Total Transactions')

    # Below we will get the total amount sold
    dates_temp_sum = df.groupby(df.Date.dt.date)['TransactionAmt'].sum().reset_index()

    # using the new dates_temp_sum we will create the second trace
    trace1 = go.Scatter(x=dates_temp_sum.Date,
                        line = dict(color = color_op[1]),
                        name="Total Amount",
                        y=dates_temp_sum['TransactionAmt'],
                        opacity = 0.8,
                        yaxis='y2')

    
    #creating the layout the will allow us to give an title and 
    # give us some interesting options to handle with the outputs of graphs
    layout = dict(
        title= "Total Transactions and Fraud Informations by Date",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible = True),
            type='date' ),
    yaxis=dict(title='Total Transactions'),
    yaxis2=dict(overlaying='y',
                anchor='x',
                side='right',
                zeroline=False,
                showgrid=False,
                title='Total Transaction Amount')
    )

    
    # creating figure with the both traces and layout
    #fig = dict(data= [trace, trace1,], layout=layout)
    fig = go.Figure(data=[trace, trace1,], layout=layout)
    
    #rendering the graphs
    fig.show()
    #iplot(fig) #it's an equivalent to plt.show()

# trans dt hour
def trans_dt_hour(df = None):

    plt.plot(df.groupby('Transaction_hour').mean()['isFraud'], color='k')
    ax = plt.gca()
    ax2 = ax.twinx()
    _ = ax2.hist(df['Transaction_hour'], alpha=0.3, bins=24, label = "Transactions")
    ax.set_xlabel('Encoded hour')
    ax.set_ylabel('Fraction of fraudulent transactions')
    ax2.set_ylabel('Number of transactions')

    plt.legend()
    plt.show()

    # trans dt hour
def card1_distr(df = None):
    bins  = np.linspace(1340,1850,50)
    
    df1 = df.groupby('card1').mean()['isFraud']
    plt.hist(df1, alpha=0.3, bins=bins, label = "Transactions", histtype = 'step')
    ax = plt.gca()
    ax2 = ax.twinx()
    _ = ax2.hist(df['card1'], alpha=0.3, bins=bins, label = "Transactions")
    ax.set_xlabel('')
    ax.set_ylabel('Fraction of fraudulent transactions')
    ax2.set_ylabel('Number of transactions')

    plt.legend()
    plt.show()

    
#id features
def id_features_display(df = None):
    #categorical features in training identity dataset
    print(
        df[['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
        'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25',
        'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32',
        'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']].describe(include='all')
    )

#id features
def id_features_distr(df = None, col = ''):
    #categorical
    
    #
    total = len(df)
        
    #nans
    #df[col] = df[col].fillna('NaN')
    df[col].fillna("NaN", inplace=True)

    #cross table
    tmp = pd.crosstab(df[col], df['isFraud'], normalize='index') * 100
    tmp = tmp.reset_index()
    tmp.rename(columns={0:'NoFraud', 1:'Fraud'}, inplace=True)

    #fig
    plt.figure(figsize=(14,10))
    plt.suptitle(f'{col} Distributions', fontsize=22)

    plt.subplot(221)
    g = sns.countplot(x=col, data=df, order=tmp[col].values)
    # plt.legend(title='Fraud', loc='upper center', labels=['No', 'Yes'])

    g.set_title(f"{col} Distribution", fontsize=19)
    g.set_xlabel(f"{col} Name", fontsize=17)
    g.set_ylabel("Count", fontsize=17)
    # g.set_ylim(0, 500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(height/total*100),
                ha="center", fontsize=14) 

    plt.subplot(222)
    g1 = sns.countplot(x=col, hue='isFraud', data=df, order=tmp[col].values)
    plt.legend(title='Fraud', loc='best', labels=['No', 'Yes'])
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y='Fraud', data=tmp, color='black', order=tmp[col].values, legend=False)
    gt.set_ylabel("% of Fraud Transactions", fontsize=16)

    g1.set_title(f"{col} by Target(isFraud)", fontsize=19)
    g1.set_xlabel(f"{col} Name", fontsize=17)
    g1.set_ylabel("Count", fontsize=17)

    plt.subplot(212)
    g3 = sns.barplot(x=col, y='TransactionAmt',
                       hue='isFraud', 
                       data=df[df['TransactionAmt'] <= 2000],
                       order=tmp[col].values )
    g3.set_title("Transaction Amount Distribution by ProductCD and Target", fontsize=20)
    g3.set_xlabel("ProductCD Name", fontsize=17)
    g3.set_ylabel("Transaction Values", fontsize=17)

    plt.subplots_adjust(hspace = 0.4, top = 0.85)

    plt.show() 

def id_features_distr_id_30(df = None):

    df.loc[df['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'
    df.loc[df['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'
    df.loc[df['id_30'].str.contains('Mac OS', na=False), 'id_30'] = 'Mac'
    df.loc[df['id_30'].str.contains('Android', na=False), 'id_30'] = 'Android'


    id_features_distr(df, 'id_30')

def id_features_distr_id_31(df = None):

    df.loc[df['id_31'].str.contains('chrome', na=False), 'id_31'] = 'Chrome'
    df.loc[df['id_31'].str.contains('firefox', na=False), 'id_31'] = 'Firefox'
    df.loc[df['id_31'].str.contains('safari', na=False), 'id_31'] = 'Safari'
    df.loc[df['id_31'].str.contains('edge', na=False), 'id_31'] = 'Edge'
    df.loc[df['id_31'].str.contains('ie', na=False), 'id_31'] = 'IE'
    df.loc[df['id_31'].str.contains('samsung', na=False), 'id_31'] = 'Samsung'
    df.loc[df['id_31'].str.contains('opera', na=False), 'id_31'] = 'Opera'
    df['id_31'].fillna("NAN", inplace=True)
    df.loc[df.id_31.isin(df.id_31.value_counts()[df.id_31.value_counts() < 200].index), 'id_31'] = "Others"

    id_features_distr(df, 'id_31')

def col_distr_id(df = None, col = '', nmax = -1):

    df.loc[df[col].str.contains('chrome', na=False), col] = 'Chrome'
    df.loc[df[col].str.contains('firefox', na=False), col] = 'Firefox'
    df.loc[df[col].str.contains('safari', na=False), col] = 'Safari'
    df.loc[df[col].str.contains('edge', na=False), col] = 'Edge'
    df.loc[df[col].str.contains('ie', na=False), col] = 'IE'
    df.loc[df[col].str.contains('samsung', na=False), col] = 'Samsung'
    df.loc[df[col].str.contains('opera', na=False), col] = 'Opera'
    df[col].fillna("NAN", inplace=True)

    col_distr(data, col, nmax)

#read in data
data = xt.read_pickle(path = "data/processed", sample = "train")

#print data
#print(data.head())

#Not all transactions have corresponding identity information
#missing_data(data)

#resume table
#print( resume(data)[:50] )

#null
#print(f'There are {data.isnull().any().sum()} columns in dataset with missing values.')

#unique values
#one_value_cols = [col for col in data.columns if data[col].nunique() <= 1]
#print(f'There are {len(one_value_cols)} columns in dataset with one unique value.')

#imbalance v
#show_imbalance(data)

#imb + stats
#show_imbalance_and_stats(data)

#target distribution
#show_target_distribution(data)

#transaction amount quantiles
#trans_amount_quantiles(data)

#transaction amount values
#trans_amount_values(data)

#transaction quantiles
#trans_quantiles(data)

#transaction outliers
#trans_outliers(data, "TransactionAmt")

#Distribution Products / Distribution of Frauds by Product
#product_cd_distributions(data)

#card features
#print(resume(data[['card1', 'card2', 'card3','card4', 'card5', 'card6']]))

#card quantiles
#card_quantiles(data)

#card distributions
card_distributions(data)
#c_features_distr(data, 'card1', 1000)
#col_distr(data, "card1", 1000)
#card_features_distr(data, 'card1', 1620)
#card1_distr(data)

#card distributions categorical
#card_categorical(data, 4, True)
#card_categorical(data, 6, True)

#explore the M features
#['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']
#M_features(data, "M2")

#address quantiles
#address_quantiles(data)

#address distributions
#col_distr(data, "addr1", 500)
#col_distr(data, "addr2", 50)

#pemailmain distributions
#P_emaildomain_distributions(data, 500)

#P_emaildomain_plot(data, "P")
#P_emaildomain_plot(data, "R")

# R email domain distr
#Emaildomain_distributions("R", data, 300)
#Emaildomain_distributions("P", data, 300)

# C features
#c_features_resume(data)

# C features distr.
#c_features_distr(data, 'C1', 400)
#c_features_distr(data, 'C2', 400)
#c_features_distr(data, 'C7', 400)

# V features
#c_features_distr(data, 'V54', 5)
#M_features(data, 'V54', 6, logy = False)

# time delta
#time_delta(data)

#Transactions and Total Amount by each day
#trans_per_day(data)

#id features
#id_features_display(data)

#id features
#'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29' 
#id_features_distr(data, 'id_32')
#id_features_distr(data, 'id_11')
#id_features_distr_id_30(data)
#id_features_distr_id_31(data)
#col_distr_id(data, "id_31", 200)

#trans dt hour
#trans_dt_hour(data)


