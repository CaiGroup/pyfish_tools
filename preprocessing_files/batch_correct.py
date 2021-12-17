from pre_processing import *
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#paths for real image
directory = Path("/groups/CaiLab/personal/Michal/raw/2021-06-21_Neuro4181_5_noGel_pool1/notebook_pyfiles/dapi_aligned")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#-----------------------------------------------------------------------
#path for background
directory = Path("/groups/CaiLab/personal/Michal/raw/2021-06-21_Neuro4181_5_noGel_pool1/initial_background")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
path_bkgrd = str(directory / position_name)

#correction function
correction_type=Gaussian_and_Gamma_Correction
swapaxes=False
stack_bkgrd=path_bkgrd
z=1
gamma=1.4
size=2048
lowpass=True

correct_many(files, correction_type, stack_bkgrd, swapaxes, z, size, gamma, lowpass)

