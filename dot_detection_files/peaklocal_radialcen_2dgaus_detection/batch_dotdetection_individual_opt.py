from dot_detection_peaklocal_rad_gaus import dot_detection_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/deconvoluted_images/")
position_name = '/MMStack_Pos0.ome.tif'

files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + position_name)
files = str(files[0])

box_size = 5 #for peak local max
exclude_border=3 #how many pixels to exclude from border
HybCycle = JOB_ID
size_cutoff = 2.5 # sigma cutoff for size distribution
channel = 3 #which channel to analyze
pos_start = 0 #referring to pos from find thresh
pos_end = 1 #referring to pos from find thresh
region_size = 7 # bounding box for gaussian fitting or radial centering
pos = None #ignore unless you are trying the visualize
choose_thresh_set = 0 #ignore for optimization
hyb_number = 12 #total number of hybs
check_initial = False #ignore for optimization
gaussian = False #do you want to gaussian fit
radial_center = True #do you want to radial center instead
optimize = True #are you testing thresholds
output = True #do you want to write out results
                 
dot_detection_parallel(files,box_size, exclude_border, HybCycle,
                       size_cutoff,channel,pos_start,pos_end,choose_thresh_set,hyb_number,region_size,
                       check_initial,gaussian, radial_center, optimize, output)