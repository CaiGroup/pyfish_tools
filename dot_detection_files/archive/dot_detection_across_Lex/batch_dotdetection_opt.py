from dot_detection_threshold_across import dot_detection_parallel
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
#only use 1 pos for now
paths_use=paths_fin[0]

min_sigma = 1.5
max_sigma = 5
num_sigma = 5
HybCycle = JOB_ID
size_cutoff=2.5
num_pos=5
num_channels=4
pos=None
choose_thresh_set = 0
check_initial = False
gaussian = True
both = False
optimize=True
output=True

dot_detection_parallel(paths_use, min_sigma, max_sigma, num_sigma, HybCycle, size_cutoff,num_pos,num_channels,pos,choose_thresh_set, check_initial, gaussian, both, optimize, output)

