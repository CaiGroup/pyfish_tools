from dot_detection_threshold_across import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/20k_dash_063021_3t3/notebook_pyfiles/deconvoluted_images/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

min_sigma = 1.5
max_sigma = 5
num_sigma = 5
HybCycle = None
size_cutoff=3
num_pos=None
num_channels=4
pos=None
choose_thresh_set = 0
check_initial = False
gaussian = True
both = False
optimize=False
output=True

dot_detection_parallel(files, min_sigma, max_sigma, num_sigma, HybCycle, size_cutoff,num_pos,num_channels,pos,choose_thresh_set, check_initial,gaussian, both, optimize, output)
