from dash_radial_decoding import radial_decoding_parallel
import os
import pandas as pd
import numpy as np
import glob

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#fill parameters
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/dots_comb/adcg/dots_comb/Pos_0/*")
codebook_path = "/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/barcode_key/codebook_string.csv"
n_neighbors = 4
num_barcodes = 4
#multiply radius by 100 to get search in nm
radius=0.50
diff=1
min_seed=3
hybs = 80
output_dir = f"/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/decoded/adcg/"
ignore_errors = False

if len(locations_path) > 1:
    for locations in locations_path:
        radial_decoding_parallel(locations, codebook_path, n_neighbors,
                                 num_barcodes, radius, diff, min_seed, hybs,
                                 output_dir, ignore_errors)
else:
    radial_decoding_parallel(locations_path[0], codebook_path, n_neighbors,
                                 num_barcodes, radius, diff, min_seed, hybs, 
                             output_dir, ignore_errors)