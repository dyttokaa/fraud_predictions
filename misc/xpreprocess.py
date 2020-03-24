import os
import pandas as pd
import xmemory as xm
import xload as xl
#pip install psutil
import psutil

path = "data"
NROWS = 50000

def mem_stat(proc=""):
    print( f'{proc} CPU percentage {psutil.cpu_percent()}')
    print( f'{proc} Physical memory {psutil.virtual_memory()}')
    print( f'{proc} Virtual memory percentage {psutil.virtual_memory()[2]}')


def load_and_reduce(ipath = "", ifile = ""):

    #df_subset = xl.get(ipath = ipath, ifile = ifile, nrows = 5000)
    #df_mem, nalist, dtype = xm.reduce_mem_usage1(df_subset)
    #print(dtype)
    #df_red =  xl.get(ipath = ipath, ifile = ifile, dtype = dtype)
    
    #return df_red
    
    df = xl.get(ipath, ifile, nrows = NROWS)
    mem_stat("Info: Before reduction of %s"%(ifile))
    df_mem, _, _ = xm.reduce_mem_usage1(df)
    del df
    mem_stat("Info: After reduction of %s"%(ifile))
    return df_mem

def reduce_and_merge(path = "", file1 = "", file2 = ""):

    df1 = load_and_reduce(path, file1)
    df2 = load_and_reduce(path, file2)

    df = pd.merge(left = df1,
                  right = df2,
                  on='TransactionID',
                  how='left',
                  #left_index=True,
                  #right_index=True
    )

    df1.to_pickle( os.path.join(path, file1.replace('csv', 'pkl')) )
    df2.to_pickle( os.path.join(path, file2.replace('csv', 'pkl')) )
    
    return df

df_train = reduce_and_merge('data', 'train_identity.csv', 'train_transaction.csv')
df_train.to_pickle( os.path.join(path, 'train.pkl') )

df_test = reduce_and_merge('data', 'test_identity.csv', 'test_transaction.csv')
df_test.to_pickle( os.path.join(path, 'test.pkl') )

df_all = pd.merge(left = df_train,
                  right = df_test,
                  on='TransactionID',
                  how='left')
df_all.to_pickle( os.path.join(path, 'all.pkl') )
