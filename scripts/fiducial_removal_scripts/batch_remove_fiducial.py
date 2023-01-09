from remove_fiducials import remove_fiducials_parallel
from glob import glob
import re
import numpy as np
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

#get paths of all dot locations for specific z and channel
directory = glob(f"/groups/CaiLab/personal/Lex/raw/120822_2k_mouse_brain/seqFISH_datapipeline/output/dots_detected/Channel_All/*/locations_z_{JOB_ID}.csv")
#sort positions to match
key = [int(re.search('Pos(\\d+)*', str(f)).group(1)) for f in directory]
locations_srcs = list(np.array(directory)[np.argsort(key)])

#get reference paths
directory = glob("/groups/CaiLab/personal/Lex/raw/120822_2k_mouse_brain/chromatic_aberration/*.ome.tif")
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

