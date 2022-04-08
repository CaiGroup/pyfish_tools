from remove_fiducials import remove_fiducials_parallel
from glob import glob
import re
import numpy as np
import os

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

#get paths of all dot locations for specific z and channel
directory = glob(f"/groups/CaiLab/personal/Lex/raw/150genes_040122/notebook_pyfiles/dots_comb/final-thresh0/Channel_{JOB_ID}/*/locations_z_0.csv")
#sort positions to match
key = [int(re.search(f'Pos(\\d+)*', str(f)).group(1)) for f in directory]
locations_srcs = list(np.array(directory)[np.argsort(key)])

#get reference paths
directory = glob("/groups/CaiLab/personal/Lex/raw/150genes_040122/Fiducials/*")
#sort positions to match
key = [int(re.search(f'Pos(\\d+)*', str(f)).group(1)) for f in directory]
fid_srcs = list(np.array(directory)[np.argsort(key)])

#parameters
threshold  = 500 #absolute threshold beads must be over
radius=1 #pixel search
swapaxes=True #bool to swapaxes


remove_fiducials_parallel(locations_srcs, fid_srcs, threshold=threshold, radius=radius, swapaxes=swapaxes)