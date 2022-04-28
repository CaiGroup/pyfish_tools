from decoding_individual import decode_individual
from pathlib import Path
import os
from webfish_tools.util import find_matching_files

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

ch = int(JOB_ID)+1

directory = Path(f"/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/notebook_pyfiles/dots_comb/Channel_{ch}")
position_name = 'MMStack_Pos0'
position_tif = 'MMStack_Pos0.ome.tif'

dot_locations = directory / position_name / 'Dot_Locations' 

data_dir='/groups/CaiLab/personal/Lex/raw/2020-08-08-takei'
position = position_tif 
decoded_dir = f'/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/notebook_pyfiles/decoded/Channel_{ch}/Pos_0' 
locations_dir = str(dot_locations)
position_dir = str(directory / position_name) 
barcode_dst = f'/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/notebook_pyfiles/decoded/Channel_{ch}/Pos_0/BarcodeKey' 
barcode_src = '/groups/CaiLab/personal/Lex/raw/2020-08-08-takei/barcode_key'

decode_individual(data_dir,position,decoded_dir, locations_dir, position_dir, 
              barcode_dst, barcode_src,bool_decoding_individual=ch,synd_decoding=False, lvf= None,zvf=None,lwvf = None)