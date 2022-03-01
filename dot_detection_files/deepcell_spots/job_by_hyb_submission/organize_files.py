"""
author: Katsuya Lex Colon
groups: CaiLab
date: 03/01/22
"""


import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def organize_files_pos(pos_path, num_z=2):
    """
    Combine the locations files after job completion when jobs are
    split by hybs. This function will combine locations in the postions
    directory.
    
    Parameters
    ----------
    pos_path: Posix path for position directory
    num_z: number of z's analyzed
    """
    
    #identify all channel directories
    channel_files = list(pos_path.glob("*"))
    #loop through channel directories and isolate specific z's
    for channel in channel_files:
        for z in range(num_z):
            comb_files = list(channel.glob(f"locations_z_{z}_*"))
            df_all = [pd.read_csv(str(df)) for df in comb_files]
            df_concat = pd.concat(df_all).reset_index(drop=True)
            df_concat.to_csv(str(channel / f"locations_z_{z}.csv"))
        files_remove = glob.glob(str(channel / "*_hyb_*.csv"))
        for file in files_remove:
            os.remove(file)

def organize_files_parallel(path, num_z=2):
    """
    Combine locations files from position directories in parallel.
    
    Parameters
    ----------
    path: general path of dot locations
    num_z: number of z's analyzed
    """
    
    import time
    start = time.time()
    
    #get all pos directories
    pos_files = list(Path(path).glob("*"))
    #parallelize by pos for combining files
    with ThreadPoolExecutor(max_workers=64) as exe:
        futures = {}
        for pos in pos_files:
            fut = exe.submit(organize_files_pos, pos, num_z)
            futures[fut] = str(pos)
        for fut in as_completed(futures):
            path = futures[fut]
            print(f'Path {path} completed after {time.time() - start} seconds')