from dapi_alignment_parallel import dapi_alignment_parallel
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#general path and position name
directory = Path("/groups/CaiLab/personal/Lex/raw/090222_150genes_acrosschannel_3t3/segmentation")
position_name = f'MMStack_Pos{JOB_ID}.ome.tif'

##get all hybs with same position
##use this for background alignment
ref = directory.parent / "chromatic_aberration" /position_name

#use this for all hyb alignment
#files, _, _ = find_matching_files(directory, 'HybCycle_{hyb}' + f'/{position_name}')
#files = [str(f) for f in files]

files = str(directory / position_name)

#get hyb 0 with same position
##use this for background alignment
# directory_align=Path("/groups/CaiLab/personal/Lex/raw/112221_20kdash_3t3/")
# ref, _, _ = find_matching_files(directory_align, 'HybCycle_0' + f'/{position_name}') 

# #use this for normal alignment
# ref, _, _ = find_matching_files(directory, 'HybCycle_0' + f'/{position_name}') 

image_ref = str(ref)
images_moving=files

dapi_alignment_parallel(image_ref,images_moving)
