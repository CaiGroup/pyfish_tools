from pre_processing import *
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#paths for real image
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/fiducial_aligned")
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
z=1
size=2048
gamma = 1.2
sigma=30
rb_radius=5
hyb_offset=0
rollingball=True
lowpass = False
match_hist=False
subtract=False
divide=True

correct_many(files, correction_type, stack_bkgrd, swapaxes, z, size, gamma, 
             sigma, rb_radius, hyb_offset, rollingball, lowpass, match_hist, subtract, divide)