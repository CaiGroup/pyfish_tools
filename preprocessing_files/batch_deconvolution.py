from pre_processing import deconvolute_many
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dapi_aligned/fiducial_aligned")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'


files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files = [str(f) for f in files]

# #for fiducial removal use this
# ref = directory / "final_background" / position_name

images = files
image_ref = None
sigma_hpgb = 30
kern_rl = 7
kern_lpgb = 3
sigma = (1.8,1.6,1.5,1.3)
radius = (3,3,3,3)
model = "gaussian"
microscope = "lb"
size = None
threshold_abs = None
gamma=1.2
hyb_offset = 0
min_distance = None
num_peaks = None
edge = None
swapaxes = False
noise = False
bkgrd_sub = False
remove_fiducial = False
match_hist=False
subtract=False
divide=True

deconvolute_many(images=images, image_ref=image_ref, 
                 sigma_hpgb=sigma_hpgb, kern_rl=kern_rl, 
                 kern_lpgb = kern_lpgb, sigma=sigma, radius=radius, model=model, microscope=microscope,
                 size=size,min_distance=min_distance,threshold_abs=threshold_abs,
                 num_peaks=num_peaks,gamma=gamma, hyb_offset=hyb_offset, edge=edge, swapaxes=swapaxes,
                 noise= noise, bkgrd_sub=bkgrd_sub, remove_fiducial=remove_fiducial,
                 match_hist=match_hist, subtract=subtract, divide=divide)