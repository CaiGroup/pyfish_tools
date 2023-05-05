from daostarfinder_dotdetection_screen import find_threshold
from pathlib import Path
import os
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')
#general image directory
directory = Path("")
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
#only use first 5 pos
paths_use=paths_fin[0:5]
#grab pos number list
pos_list = []
for path in paths_use:
    pos_num = Path(path).name.split("_")[1].split(".")[0].replace("Pos","")
    pos_list.append(int(pos_num))

#parameters
threshold_min  = 0.01 #starting minimum threshold 
threshold_max = 0.7 #ending maximum threshold
interval = 100 #interval between min and max
HybCycle = JOB_ID #JOB id from slurm task array
channel = 1 #which channel (1-4)
reduce_cutoff = 4 #number of indexes to go back from sliding window (2 or 4 is good)
window=5 #size of sliding window (5 if interval is 100 and 10 if interval is 200)

find_threshold(paths_use,threshold_min=threshold_min,
               threshold_max=threshold_max, interval=interval, HybCycle=HybCycle, 
               channel=channel, pos_list=pos_list,reduce_cutoff=reduce_cutoff, window=window)
