from svm_feature_radial_decoding import feature_radial_decoding
import os
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/seqFISH_datapipeline/output/dots_detected/Channel_All/Pos{JOB_ID}/*")
#general codebook path
codebook_path = f"/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/barcode_key/codebook_string_across.csv"
#number of readout sites
num_barcodes = 4
#search radii
first_radius = 1
second_radius = 1.5
third_radius = 2
#how many allowed drops in calls 
diff = 1
#how many times does a pseudocolor sequence must appear
min_seed = 3
#how many times does pseudocolor sequence must appear for highly expressed genes
high_exp_seed = 3
#number of total hybs
hybs = 8
#probability cutoff for On dots (0-1). Lower the value the less stringent. Setting probability_cutoff=0 and desired_fdr=None, will output normal unfiltered data.
probability_cutoff = 0.15
#desired FDR (0-1). Could set to None if you would like to filter yourself.
desired_fdr = 0.10
#do you have parity round
parity_round = True
#do you want locations of dots that didn't pass parity
include_undefined = False
#do you want to decode highly expressed genes first
decode_high_exp_genes = True
#do you want to perform an additional third round of decoding
triple_decode = True
#Where do you want to output the files
output_dir = f"/groups/CaiLab/personal/Lex/raw/092222_150_nih_3t3/seqFISH_datapipeline/output/decoded/final_11p52_33_heg_svm_0p15_diff1_fdr5/Channel_All/Pos_{JOB_ID}"

if len(locations_path) > 1:
    for locations in locations_path:
        feature_radial_decoding(location_path=locations, codebook_path=codebook_path,
                             num_barcodes=num_barcodes, first_radius=first_radius, 
                             second_radius=second_radius,third_radius=third_radius,
                             diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, hybs=hybs, 
                             probability_cutoff=probability_cutoff, desired_fdr = desired_fdr, 
                             output_dir=output_dir, include_undefined=include_undefined, 
                             decode_high_exp_genes=decode_high_exp_genes,
                             triple_decode=triple_decode, parity_round=parity_round)
else:
    feature_radial_decoding(location_path=locations_path[0], codebook_path=codebook_path,
                         num_barcodes=num_barcodes, first_radius=first_radius, second_radius=second_radius,
                         third_radius=third_radius, diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, hybs=hybs, 
                         probability_cutoff=probability_cutoff,desired_fdr = desired_fdr,
                         output_dir=output_dir, include_undefined=include_undefined,
                         decode_high_exp_genes=decode_high_exp_genes,
                         triple_decode=triple_decode, parity_round=parity_round)

