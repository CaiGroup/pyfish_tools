"""
author: Katsuya Lex Colon
groups: CaiLab
date: 03/01/22
"""

import os.path
import pandas as pd
from pathlib import Path
import numpy as np
import glob
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_hyb_files(paths, num_hybs):
    """
    Check to see if number of files match the number of hybs
    Parameters
    ----------
    paths = list of all hyb paths 
    num_hybs = integer for number of total hybs
    """
    #check if number of hybs match number of files
    if len(paths) != num_hybs:
        hybs_present = []
        #go through paths and see which hyb is present
        for path in paths:
            hyb = path.name.split("_")[4].replace(".csv","")
            hybs_present.append(int(hyb))
        #generate list of all hybs
        all_hybs = np.arange(0,num_hybs,1).tolist()
        #check which is missing
        missing_hybs = set(all_hybs) - set(hybs_present)
        
        return missing_hybs
    else:
        return []
    
def organize_files_pos(pos_path, num_z=2, num_hybs=80):
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
            #check if all files are present
            if check_hyb_files(comb_files, num_hybs) == []:
                df_all = [pd.read_csv(str(df)) for df in comb_files]
                df_concat = pd.concat(df_all).reset_index(drop=True)
                df_concat.to_csv(str(channel / f"locations_z_{z}.csv"))
                
            else:
                missing_hybs = check_hyb_files(comb_files, num_hybs)
                position = pos_path.name
                channel_num = channel.name
                if os.path.isfile("missing_hybs.txt"):
                    with open("missing_hybs.txt","a+") as f:
                        f.write(f"{channel_num}, {position}, z= {z}, hybs missing = {missing_hybs}\n")
                else:
                    with open("missing_hybs.txt","w+") as f:
                        f.write(f"{channel_num}, {position}, z= {z}, hybs missing = {missing_hybs}\n")
        files_remove = glob.glob(str(channel / "*_hyb_*.csv"))
        for file in files_remove:
            os.remove(file)
                    
    
def organize_files_parallel(path, num_z=2, num_hybs=80):
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
        for pos in pos_files:
            fut = exe.submit(organize_files_pos, pos, num_z, num_hybs)
