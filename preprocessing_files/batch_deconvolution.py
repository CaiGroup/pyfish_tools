from pre_processing import deconvolute_many
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to images
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/fiducial_aligned")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

# #for fiducial removal use this
# ref = directory / "final_background" / position_name

images = files #images
sigma_hpgb = 30 #sigma for high pass gaussian blur
kern_hpgb = 5 #kernel size for 2d gaussian blur
kern_rl = 7 #kernel size for richardson-lucy deconvolution
kern_lpgb = 3 #kernel size for low pass gaussian
sigma = (1.8,1.6,1.5,1.3) #sigma values gaussin psf for each channel in richardson-lucy deconvolution
radius = (3,3,3,3) #radius for airy disc psf for each channel in richardson-lucy deconvolution
model = "gaussian" #define which psf model to use(gaussian or airy disc)
microscope = "lb" #use pre-defined sigma psf for lb (leica boss) or boc (box of chocolates)
gamma=1.2 #how much do you want to gamma enhance (1 will do no enhancment)
hyb_offset = 0 #set certain hybcycle to 0
swapaxes = False #set to true if your image is c,z,x,y
noise = False #add gaussian noise back
bkgrd_sub = False #bool to perform background subtraction
match_hist=False #bool to enhance gaussian blurred image for high pass filter
subtract=False #bool to perform high pass filter
divide=True #bool to even out illumination 
tophat_raw=False#bool to perform tophat on raw image before any other preprocessing steps

deconvolute_many(images=images, 
                 sigma_hpgb=sigma_hpgb, kern_hpgb=kern_hpgb, kern_rl=kern_rl, 
                 kern_lpgb = kern_lpgb, sigma=sigma, radius=radius, model=model, microscope=microscope,
                 gamma=gamma, hyb_offset=hyb_offset, swapaxes=swapaxes,
                 noise= noise, bkgrd_sub=bkgrd_sub,
                 match_hist=match_hist, subtract=subtract, divide=divide, tophat_raw=tophat_raw)
