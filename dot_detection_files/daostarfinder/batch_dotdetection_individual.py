from daostarfinder_dotdetection import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/pre_processed_images/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

HybCycle = JOB_ID
size_cutoff = 3 # sigma cutoff for size distribution
channel = 4 #which channel to analyze
pos_start = 0 #referring to pos from find thresh
pos_end = 3 #referring to pos from find thresh (exclusive)
choose_thresh_set = 3 #select best thresh set
hyb_number=12 #total number of hybs
check_initial = False #ignore unless you are trying to visualize
optimize=False #are you testing thresholds
output=True #do you want to write out results

dot_detection_parallel(files, HybCycle,size_cutoff,channel,pos_start,pos_end,
                       choose_thresh_set,hyb_number, check_initial, optimize, output)
