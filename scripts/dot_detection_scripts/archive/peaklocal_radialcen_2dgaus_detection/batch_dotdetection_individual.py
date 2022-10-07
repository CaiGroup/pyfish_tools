from dot_detection_peaklocal_rad_gaus import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/pre_processed_images/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

box_size = 5 #for peak local max
exclude_border = 3 #how many pixels to exclude from border
HybCycle = JOB_ID
size_cutoff = 3 # sigma cutoff for size distribution
channel = 3 #which channel to analyze
pos_start = None #ignore if not optimizing
pos_end = None #ignore if not optimizing
region_size=7 # bounding box for gaussian fitting or radial centering
pos=None #ignore unless you are trying the visualize
choose_thresh_set = 6 #select best thresh set
hyb_number=12 #total number of hybs
check_initial = False #ignore unless you are trying to visualize
gaussian = False #do you want to gaussian fit
radial_center = True #do you want to radial center instead
optimize=False #are you testing thresholds
output=True #do you want to write out results

dot_detection_parallel(files, box_size, exclude_border, HybCycle,
                       size_cutoff,channel,pos_start,pos_end,choose_thresh_set,hyb_number,region_size,
                       check_initial,gaussian, radial_center, optimize, output)
