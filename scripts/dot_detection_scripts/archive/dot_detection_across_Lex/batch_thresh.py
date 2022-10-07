from dot_detection_threshold_across import find_threshold
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')
#general image directory
directory = Path("/groups/CaiLab/personal/Lex/raw/20k_dash_062421_brain/notebook_pyfiles/deconvoluted_images/")
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

min_sigma = 1.5
max_sigma = 5
num_sigma = 5
threshold_min  = 0.002
threshold_max = 0.015
interval = 100
HybCycle = JOB_ID
channel = "all"
min_dots_start = 10000 
min_dots_end = 100000
num_pos = len(paths_use)
strict = True

find_threshold(paths_use, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold_min=threshold_min,
               threshold_max=threshold_max, interval=interval, HybCycle=HybCycle, 
               channel=channel, min_dots_start=min_dots_start, min_dots_end=min_dots_end,num_pos=num_pos, strict=strict)