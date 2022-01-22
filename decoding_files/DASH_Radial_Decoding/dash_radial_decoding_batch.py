from dash_radial_decoding import dash_radial_decoding
import os
import pandas as pd
import numpy as np
import glob

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#fill parameters
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dots_comb/Channel_1/MMStack_Pos0/Threshold_{JOB_ID}/Dot_Locations/*")
codebook_path = "/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/barcode_key/codebook_string.csv"
n_neighbors = 4
num_barcodes = 4
#multiply radius by 100 to get search in nm
first_radius=1
second_radius=2
diff=1
min_seed=3
hybs = 80
include_undecoded = False
triple_decode = True
output_dir = f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/decoded/final_lowthresh/MMStack_Pos{JOB_ID}"

if len(locations_path) > 1:
    for locations in locations_path:
        dash_radial_decoding(locations, codebook_path, n_neighbors,
                             num_barcodes, first_radius, second_radius,
                             diff, min_seed, hybs, 
                             output_dir, include_undecoded, triple_decode)
else:
    dash_radial_decoding(locations_path[0], codebook_path, n_neighbors,
                         num_barcodes, first_radius, second_radius,
                         diff, min_seed, hybs, 
                         output_dir, include_undecoded, triple_decode)
    