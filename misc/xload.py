import os
import pandas as pd
import numpy as np
from pathlib import Path

def join(ipath = "", ifile = ""):
    return os.path.join(ipath, ifile)

def get(ipath = "", ifile="", nrows = None, dtype = None):
    df =  pd.read_csv(filepath_or_buffer = join(ipath, ifile),
                      sep = ",",
                      error_bad_lines = False,
                      engine = "c",
                      nrows = nrows,
                      dtype = dtype,
                      #usecols = [0,1,2]
                      #low_memory=False
    )

    print(f'Test dataset = {ifile}. Rows = {df.shape[0]}. Columns = {df.shape[1]}.')
    return df


def read_pickle(path = "", sample = ""):

    pkl = join(path, sample + ".pkl")
    if Path( pkl ).is_file():
        return pd.read_pickle( pkl )
    else:
        print(f'Error: {pkl} not found')
        return None
    
    
def getData(path = "", train = False, test = False):
    
    if train: 
        train_identity = pd.read_csv(filepath_or_buffer = join(path, 'train_identity.csv'),
                                     sep = ",",
                                     error_bad_lines = False,
                                     #low_memory=False
        )

        train_transaction = pd.read_csv(filepath_or_buffer = join(path, 'train_transaction.csv'),
                                        sep = ",",
                                        error_bad_lines = False,
                                        #low_memory=False
        )

        train = pd.merge(left = train_transaction,
                         right = train_identity,
                         on='TransactionID',
                         how='left')

        print(f'Train dataset. Rows = {train.shape[0]}. Columns = {train.shape[1]}.')

    if test:
        test_identity = pd.read_csv(filepath_or_buffer = join(path, 'test_identity.csv'),
                                    sep = ",",
                                    error_bad_lines = False,
                                    #low_memory=False
        )

        test_transaction = pd.read_csv(filepath_or_buffer = join(path, 'test_transaction.csv'),
                                       sep = ",",
                                       error_bad_lines = False,
                                       #low_memory=False
        )

        test = pd.merge(left = test_transaction,
                        right = test_identity,
                        on='TransactionID',
                        how='left')

        print(f'Test dataset. Rows = {test.shape[0]}. Columns = {test.shape[1]}.')
        
    if train and test:
        total = pd.concat( [test, train] )
    
    if train and not test:
        return train
    elif not train and test:
        return test
    elif train and test:
        return total
    else:
        print("Warning! No samples to be loaded.")
        return None
    
