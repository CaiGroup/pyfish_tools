from daostarfinder_dotdetection import find_threshold
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')
#general image directory
directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/pre_processed_images/")
#jobs will be split by hybs
hybcycle = f"HybCycle_{JOB_ID}"
#gen path with hyb
path = directory/hybcycle
#get all pos for hyb
path_pos= list(path.glob("*.tif"))
files = [str(f) for f in path_pos]
#organize paths numerically
key = [int(re.search('MMStack_Pos(\\d+)', f).group(1)) for f in files]
paths_fin = list(np.array(files)[np.argsort(key)])
#only use first 3 pos
paths_use=paths_fin[0:3]

threshold_min  = 100 #starting minimum threshold 
threshold_max = 1500 #ending maximum threshold
interval = 100 #interval between min and max
HybCycle = JOB_ID #JOB id from slurm task array
channel = 4 #which channel (1-4)
pos_start = 0 #pos number start range for getting thresholds
pos_end = 3 #pos number end range for getting thresholds
reduce_cutoff = 4 #number of indexes to go back from sliding window (2 or 4 is good)
window=5 #size of sliding window (5 if interval is 100 and 10 if interval is 200)

find_threshold(paths_use,threshold_min=threshold_min,
               threshold_max=threshold_max, interval=interval, HybCycle=HybCycle, 
               channel=channel, pos_start=pos_start,pos_end=pos_end,reduce_cutoff=reduce_cutoff, window=window)
