from pre_processing import *
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#paths for real image
directory = Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/notebook_pyfiles/aberration_corrected")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#-----------------------------------------------------------------------
# #path for background
# directory = Path("/groups/CaiLab/personal/Lex/raw/576_readout_screen/100621_576readout_screen/notebook_pyfiles/dapi_aligned/final_background")
# position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
# path_bkgrd = str(directory / position_name)

#correction function and arguments
correction_type=Gaussian_and_Gamma_Correction
swapaxes=False
stack_bkgrd=None
z=2
size=2048
gamma = 1.4
sigma=20
rb_radius=5
rollingball=True
lowpass = True

correct_many(files, correction_type, stack_bkgrd, swapaxes, z, size, gamma, sigma, rb_radius, rollingball, lowpass)