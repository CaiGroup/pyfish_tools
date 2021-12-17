from daostarfinder_detect import find_threshold
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
paths_use=paths_fin[0:3]

sigma_radius=2
roundlo=0.5
roundhi=1
brightest=None
threshold_min  = 100
threshold_max = 3000
interval = 100
HybCycle = JOB_ID
channel = "all"
min_dots_start = 3000 
min_dots_end = 100000
num_pos = len(paths_use)
strict = True

find_threshold(paths_use, sigma_radius=sigma_radius, roundlo=roundlo,
               roundhi=roundhi,brightest=brightest,threshold_min=threshold_min,
               threshold_max=threshold_max,interval=interval, HybCycle=HybCycle, 
               channel=channel, min_dots_start=min_dots_start, min_dots_end=min_dots_end,num_pos=num_pos, strict=strict)