from dash_radial_decoding import dash_radial_decoding
import os
import pandas as pd
import numpy as np
import glob

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#fill parameters
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/notebook_pyfiles/dots_detected/Pos0/*")
# locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/notebook_pyfiles/dots_comb/channels_combined/Threshold_{JOB_ID}/*")
codebook_path = "/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/barcode_key/corrected_codebook_4channel_12hybs.csv"
num_barcodes = 4
#multiply radius by 100 to get search in nm
first_radius = 1.5
second_radius = 2
third_radius = 3
diff=0
min_seed=4
hybs = 12
include_undecoded = False
triple_decode = True
decode_across = True
output_dir = f"/groups/CaiLab/personal/Lex/raw/020422_20kdash_3t3/notebook_pyfiles/decoded/deepcell/seed_4/"

if len(locations_path) > 1:
    for locations in locations_path:
        dash_radial_decoding(locations, codebook_path,
                             num_barcodes, first_radius, second_radius,third_radius,
                             diff, min_seed, hybs, 
                             output_dir, include_undecoded, triple_decode, decode_across)
else:
    dash_radial_decoding(locations_path[0], codebook_path,
                         num_barcodes, first_radius, second_radius,third_radius,
                         diff, min_seed, hybs, 
                         output_dir, include_undecoded, triple_decode, decode_across)
    