from dapi_alignment_parallel import dapi_alignment_parallel
from pathlib import Path
import os
#enable relative import
import sys 
sys.path.append("..")
#custom py files
from helpers.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory     = Path("/groups/CaiLab/personal/Lex/raw/230608_4k_inv_5bs/")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

#set reference positions
ref = directory / "chromatic_aberration" / position_name

#use this for all hyb alignment
files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
files       = [str(f) for f in files]

# #use this for single images
# files = str(directory / position_name)

image_ref     = str(ref)
images_moving = files
num_channels  = 4

dapi_alignment_parallel(image_ref,images_moving, num_channels)
