from decoding_across import decode_across
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
#30 min for 168 pos

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

directory = Path("/groups/CaiLab/personal/Lex/raw/20k_dash_063021_3t3/notebook_pyfiles/dots_comb/final")
position_name = f'MMStack_Pos{JOB_ID}'
position_tif = f'MMStack_Pos{JOB_ID}.ome.tif'

dot_locations = directory / position_name / 'Dot_Locations' 

data_dir='/groups/CaiLab/personal/Lex/raw/20k_dash_063021_3t3'
position = position_tif 
decoded_dir = f'/groups/CaiLab/personal/Lex/raw/20k_dash_063021_3t3/notebook_pyfiles/decoded/Pos_{JOB_ID}' 
locations_dir = str(dot_locations)
position_dir = str(directory / position_name) 
barcode_dst = f'/groups/CaiLab/personal/Lex/raw/20k_dash_063021_3t3/notebook_pyfiles/decoded/Pos_{JOB_ID}/BarcodeKey' 
barcode_src = '/groups/CaiLab/personal/Lex/raw/20k_dash_063021_3t3/barcode_key'

decode_across(data_dir,position,decoded_dir, locations_dir, position_dir, 
              barcode_dst, barcode_src,synd_decoding=False, lvf= None,zvf=None,lwvf = None)