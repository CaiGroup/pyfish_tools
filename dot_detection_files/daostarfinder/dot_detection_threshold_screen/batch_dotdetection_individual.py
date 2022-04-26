from daostarfinder_dotdetection_screen import dot_detection_parallel
from pathlib import Path
import glob
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')
#path to processed images
directory = Path("/groups/CaiLab/personal/Lex/raw/150genes3bind_040622/notebook_pyfiles/pre_processed_images/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

HybCycle = None #leave this to none if optiomization is false
pos_list = [0] #include one of the positions used in threshold screen (used to collect FWHM)
size_cutoff = 4 # sigma cutoff for size distribution
channel = 1 #which channel to analyze
choose_thresh_set = 7 #select best thresh set
hyb_number=18 #total number of hybs
optimize=False #are you testing thresholds
output=True #do you want to write out results

dot_detection_parallel(files, HybCycle,size_cutoff,channel,pos_list,
                       choose_thresh_set,hyb_number, optimize, output)
