import pandas as pd 
import pickle
import glob
import xmemory as xm

def chunkData(in_path = "", #Path where the large file is
              out_path = "", #Path to save the pickle files to
              chunk_size = 10000, #size of chunks relies on your available memory
              separator = ","):
    print("read in the data as chunks and save each chunk as pickle")

    reader = pd.read_csv(in_path,
                         sep=separator,
                         chunksize=chunk_size, 
                         low_memory=False)    
    
    for i, chunk in enumerate(reader):
        out_file = out_path + "/data_{}.pkl".format(i+1)
        with open(out_file, "wb") as f:
            pickle.dump(chunk,f,pickle.HIGHEST_PROTOCOL)


def joinData(pickle_path = ""): #Same Path as out_path i.e. where the pickle files are
    print("read in the pickles and append each pickle to your desired dataframe")
    
    data_p_files=[]
    for name in glob.glob(pickle_path + "/data_*.pkl"):
        data_p_files.append(name)


    df = pd.DataFrame([])
    for i in range(len(data_p_files)):
        print(f"Appending pkl {i}/{len(data_p_files)}")

        df_chunk = pd.read_pickle(data_p_files[i])
        df_chunk_mem = xm.reduce_mem_usage2(df_chunk)
                
        df = df.append(df_chunk_mem,
                       ignore_index=True)

        del df_chunk_mem
        
chunkData( "data/train_transaction.csv", "chunks" )
joinData("chunks")
