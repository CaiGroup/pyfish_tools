from pathlib import Path
import os
import sys
from util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

#from daostarfinder_dotdetection import dot_detection_parallel

# # get channel info
# channel = int(sys.argv[1])

# print(f'This is task {JOB_ID}')
## path to processed images
# directory = Path("/groups/CaiLab/personal/Lex/raw/231013_3k_9primers_12probes_5nM/pyfish_tools/output/pre_processed_images_lp")
# position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

# files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
# files = [str(f) for f in files]

# threshold    = 0.01 # threshold to detect all dots (if image was scaled)
# num_channels = 4 #number of channels in image
# channel = channel #which channel to analyze
# size_cutoff = None # sigma cutoff for size distribution

# dot_detection_parallel(img_src = files, size_cutoff=size_cutoff, threshold=threshold, channel=channel, num_channels=num_channels)

#====================================================================================================================

from dot_detection_rad import dot_detection_radial_center

print(f'This is task {JOB_ID}')
# path to processed images
directory = Path("/groups/CaiLab/personal/Lex/raw/231013_3k_9primers_12probes_5nM/pyfish_tools/output/pre_processed_images_lp")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

threshold    = 0.01 # threshold to detect all dots (if image was scaled)
num_channels = 4 #number of channels in image

dot_detection_radial_center(img_src=files, threshold=threshold, num_channels=num_channels)