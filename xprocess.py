import pandas as pd
import numpy as np
import xmemory as xm
#pip install psutil
import psutil, os
import time
import xfeatures as xf
import gc

outpath = "data/processed"

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
        df[c + '_bin']    = df[c].map(emails)
        df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
        df[c + '_suffix'] = df[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

        
def transaction_dt(df = None):
        # https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
        df['Transaction_dow'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7) #Creates a day of the week feature, encoded as 0-6. 
        df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24  # Creates an hour of the day feature, encoded as 0-23. 

def fill_nan(df = None):
    """
    Options:
    fillna(value=df_train.mean(), inplace=True)
    """
    columns             = df.columns
    numeric_columns     = df._get_numeric_data().columns
    categorical_columns = list(set(columns) - set(numeric_columns))


    print('Fill NaN - categorical columns:', sorted(categorical_columns))

    # categorical fill NaN
    #df[categorical_columns].replace({ np.nan:'missing'}, inplace = True)
    df[categorical_columns] = df[categorical_columns].replace({ np.nan:'missing'})

    print('Fill NaN - numeric columns:', sorted(numeric_columns))
        
    # categorical fill NaN
    #df[numeric_columns].replace(np.nan, 0, inplace = True)
    df[numeric_columns] = df[numeric_columns].fillna(value=0.)


    print("Number of all columns", len(columns))
    print("Number of categorical columns", len(categorical_columns))
    print("Number of numeric columns", len(numeric_columns))
        
def lbl_encode(df = None, todrop = [] ):
    #transform / label encoding

        
    columns             = df.columns
    numeric_columns     = df._get_numeric_data().columns
    categorical_columns = list(set(columns) - set(numeric_columns))

    columns_list = df.drop(todrop, axis=1).columns if todrop else df.columns  
    for f in columns_list:
        if df[f].dtype=='object' or df[f].dtype=='object':         
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[f].values) + list(df[f].values))
            df[f] = lbl.transform(list(df[f].values))

    print("Encoded categorical columns", df[categorical_columns].head)
    
def readin(sfile = '', index_col = '', columns = None):

    df = pd.read_csv(filepath_or_buffer = sfile,
                           index_col=index_col,
                           skip_blank_lines=True,
                           #low_memory=False,
                           usecols = columns
    )

    print("File = {}".format(sfile))
    print("Shape = {:,} rows, {:,} columns".format(df.shape[0], df.shape[1]))
    print("Memory usage = {:.2f}GB".format(df.memory_usage().sum() / 1024**3))
    return df

from sklearn import preprocessing
def process(sample = "", columns_trans = [], columns_id = [], isTrain = False):


    ################
    ### Read-in data
    ################
    start = time.time()
    df_trans = readin('data/%s_transaction.csv'%(sample), 'TransactionID', columns_trans)
    df_id = readin('data/%s_identity.csv'%(sample), 'TransactionID', columns_id)
    end = time.time()
    print("\nLoad in memory: %.1f sec"%(end-start))    

    """
    pd.read_csv(filepath_or_buffer = 'data/%s_transaction.csv'%(sample),
                           index_col='TransactionID',
                           skip_blank_lines=True,
                           #low_memory=False,
                           usecols = columns
    )

    df_id = pd.read_csv(filepath_or_buffer = 'data/%s_identity.csv'%(sample),
                        index_col='TransactionID',
                        skip_blank_lines=True
    )
    """
    ################
    ### Merge data
    ################

    start = time.time()
    df_merge = pd.merge(left = df_trans,
                        right = df_id,
                        on='TransactionID',
                        how='left'
    )
    end = time.time()
    print("\nMerge: %.1f sec"%(end-start))
    
    print(f'{sample} dataset. Rows = {df_merge.shape[0]}. Columns = {df_merge.shape[1]}.')

    print("Number of columns", len(df_merge.columns))
    
    df_reduce, _, _ = xm.reduce_mem_usage1(df_merge)

    del df_merge
    gc.collect()
    print('GC done')

    ################
    ### Map emails
    ################
    print("\nEmail mapping: shape before")    
    print(df_reduce.shape)

    start = time.time()
    steer_emails(df_reduce)
    end = time.time()
    print("Email mapping: %.1f sec"%(end-start))    

    print("Email mapping: shape after")
    print(df_reduce.shape)
    print(f"Email maming: values {df_reduce.columns.values}")
    print(df_reduce['P_emaildomain_bin'].head())
    print(df_reduce['P_emaildomain_suffix'].head())

    ################
    ### Trans dT
    ################
    print("\ndT Trans: shape before")    
    print(df_reduce.shape)

    start = time.time()
    transaction_dt(df_reduce)
    end = time.time()
    print("dT Trans: %.1f sec"%(end-start))    

    print("dT trans: shape after")
    print(df_reduce.shape)
    print(df_reduce['Transaction_dow'].head())
    print(df_reduce['Transaction_hour'].head())

    ################
    ### Fill NaN
    ################
    print("\nFill NaN: shape before")    
    print(df_reduce.shape)

    na_columns = df_reduce.isna().sum()
    print( "Columns fraction with NaN before\n", na_columns[na_columns>0]/df_reduce.shape[0])

    start = time.time()
    fill_nan(df_reduce)
    end = time.time()
    print("Fill NaN: %.1f sec"%(end-start))    

    na_columns = df_reduce.isna().sum()
    print( "Columns fraction with NaN after\n", na_columns[na_columns>0]/df_reduce.shape[0])

    print("Fill NaN: shape after")
    print(df_reduce.shape)


    ##################
    ### Label encoding
    ##################
    print("\nLE: shape before")    
    print(df_reduce.shape)

    start = time.time()
    coldrop = ['isFraud'] if isTrain else []
    lbl_encode(df_reduce, coldrop)
    end = time.time()
    print("LE: %.1f sec"%(end-start))    

    print("LE: shape after")
    print(df_reduce.shape)
    #print(df_reduce['Transaction_dow'].head())
    #print(df_reduce['Transaction_hour'].head())



    
    # Cleaning infinite values to NaN
    # by https://www.kaggle.com/dimartinot
    #start = time.time()
    #df_reduce.replace([np.inf, -np.inf], np.nan, inplace=True)
    #end = time.time()
    #print("Cleaning data: %.1f sec"%(end-start)) 


    ################
    ### Store
    ################
    outfile = os.path.join(outpath, f'{sample}.pkl')
    df_reduce.to_pickle( outfile )
    print(f'{outfile} created...')

    outfile2 = os.path.join(outpath, f'{sample}.txt')
    df_reduce.to_csv( outfile2, index = False )
    print(f'{outfile2} created...')
    
    
#train
process(sample = "train",
        columns_trans = xf.columns_trans,
        columns_id = xf.columns_id,
        isTrain = True)

#test
columns_trans = xf.columns_trans
columns_trans.remove('isFraud')
process(sample = "test",
        columns_trans = columns_trans,
        columns_id = xf.columns_id)



