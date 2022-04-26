from daostarfinder_dotdetection import dot_detection_parallel
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

size_cutoff = 4 # sigma cutoff for size distribution
threshold = 0.02 # threshold to detect all dots
channel = 1 #which channel to analyze

dot_detection_parallel(img_src = files,size_cutoff=size_cutoff, threshold=threshold, channel=channel)
