from ratiometric_dot_detection import ratiometric_dot_detection_parallel
from pathlib import Path
import os
from util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')
#path to processed images
directory = Path("/groups/CaiLab/personal/Lex/raw/Ratio_metric_Lantern/05312022_automation/notebook_pyfiles/pre_processed_images/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

size_cutoff = None # sigma cutoff for size distribution
threshold = 500 # threshold to detect all dots (if image was scaled)
radius = 3 # radius search
pixel_based = False #do you want to do spot based or pixel based dot detection
num_channels=4 #number of channels in image

ratiometric_dot_detection_parallel(img_src = files, size_cutoff = size_cutoff, 
                                   threshold = threshold, radius = radius,
                                   pixel_based = pixel_based,  num_channels = num_channels)


