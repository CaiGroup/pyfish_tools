from pre_processing import *
from pathlib import Path
import os
from util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#paths for real image
directory = Path("")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

#-----------------------------------------------------------------------
# #path for background
# directory = Path("")
# position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
# path_bkgrd = str(directory / position_name)


correction_type=Gaussian_and_Gamma_Correction #correction function 
num_channels=4 #number of channels in your image
stack_bkgrd=None #path to background images
gamma = 1.0 #gamma enhancement
kern_hpgb=7 #kernel size for 2d gaussian blurring
sigma=10 #sigma for 2d or 1d gaussian blurring depending on argument combination
rb_radius=5 #rolling ball radius
hyb_offset=0 #this value is used to set a certain hyb as 0
p_min=80 #minimum percentile for intensity clipping
p_max = 99.999 #maximum percentile for intensity clipping
norm_int = True #bool to scale image
rollingball=False #bool to perform rolling ball
lowpass = True #bool to perform 2d lowpass gaussian blurring
match_hist=False #bool to boost background blurred image for high pass gaussian
subtract=True #bool to perform high pass gaussian filter
divide=False #bool to even out illumination
tophat_raw=False#bool to perform tophat on raw image before any other preprocessing steps

correct_many(files, correction_type = correction_type, stack_bkgrd=stack_bkgrd, num_channels=num_channels, 
             gamma=gamma, kern_hpgb=kern_hpgb,
             sigma=sigma, rb_radius=rb_radius, p_min=p_min, p_max=p_max, norm_int=norm_int, 
             hyb_offset=hyb_offset, rollingball=rollingball, lowpass=lowpass, match_hist=match_hist, 
             subtract=subtract, divide=divide, tophat_raw=tophat_raw)

