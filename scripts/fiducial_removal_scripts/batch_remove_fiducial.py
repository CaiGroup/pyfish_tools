from remove_fiducials import remove_fiducials_parallel
from glob import glob
import re
import numpy as np
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

#get paths of all dot locations for z and channel
directory = glob(f"/groups/CaiLab/personal/Lex/raw/230521_10k_human_AD/pyfish_tools/output/dots_detected/Channel_All/*/locations_z_{JOB_ID}.csv")
#sort positions to match
key = [int(re.search('Pos(\\d+)*', str(f)).group(1)) for f in directory]
locations_srcs = list(np.array(directory)[np.argsort(key)])

#get reference paths
directory = glob(f"/groups/CaiLab/personal/Lex/raw/230521_10k_human_AD/pyfish_tools/output/pre_processed_images/chromatic_aberration/*)
#sort positions to match
key = [int(re.search('Pos(\\d+)*', str(f)).group(1)) for f in directory]
fid_srcs = list(np.array(directory)[np.argsort(key)])

#parameters
threshold  = 1000 #absolute threshold fiducials must be over
radius = 2 #pixel search
num_channels = 4 # number of channel in image
max_project = True #max project fiducial image

remove_fiducials_parallel(locations_srcs, fid_srcs, threshold=threshold, 
                          radius=radius, num_channels=num_channels, max_project = max_project)

