from dot_detection_blob_rad_gaus import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/notebook_pyfiles/sliced_img")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

min_sigma = 1.5
max_sigma = 5
num_sigma = 5
HybCycle = JOB_ID
size_cutoff=3
channel=0
pos_start=0
pos_end=1
region_size=7
pos=None
choose_thresh_set = 0
hyb_number=65
check_initial = False
gaussian = True
radial_center = False
optimize=False
output=True

dot_detection_parallel(files, min_sigma, max_sigma, num_sigma, HybCycle,
                       size_cutoff,channel,pos_start,pos_end,choose_thresh_set,hyb_number,region_size,
                       check_initial,gaussian, radial_center, optimize, output)
