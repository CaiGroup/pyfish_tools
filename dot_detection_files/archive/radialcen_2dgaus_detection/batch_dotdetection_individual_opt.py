from dot_detection_blob_rad_gaus import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/deconvoluted_images")
position_name = '/MMStack_Pos0.ome.tif'

files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + position_name)
files = str(files[0])

min_sigma = 1.5
max_sigma = 5
num_sigma = 5
HybCycle = JOB_ID
size_cutoff=2
channel=0
pos_start=0
pos_end=1
region_size=7
pos=None
choose_thresh_set = 0
hyb_number=80
check_initial = False
gaussian = False
radial_center = True
optimize=True
output=True

dot_detection_parallel(files, min_sigma, max_sigma, num_sigma, HybCycle,
                       size_cutoff,channel,pos_start,pos_end,choose_thresh_set,hyb_number,region_size,
                       check_initial,gaussian, radial_center, optimize, output)
