from svm_feature_radial_decoding import feature_radial_decoding
import os
import glob
import sys

JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)

print(f'This is task {JOB_ID}')

#name of codebooks
codebooks = ["codebook_string_647.csv","codebook_string_561.csv","codebook_string_488.csv"]

#collect pos and channel info
#pos = int(sys.argv[1])
channel = int(sys.argv[2])

#path to dots
locations_path = glob.glob(f"/groups/CaiLab/personal/Lex/raw/06082022_4kgenes/seqFISH_datapipeline/dots_detected/Channel_{channel}/genes_in_cells/Pos{JOB_ID}/locations_z_0.csv")
#general codebook path
codebook_path = f"/groups/CaiLab/personal/Lex/raw/06082022_4kgenes/seqFISH_datapipeline/decoding_files/SVM_Feature_Radial_Decoding/codebook_converter/{codebooks[channel-1]}"
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
hybs = 48
#probability cutoff for On dots (0-1). Lower the value the less stringent. Setting probability_cutoff=0 and desired_fdr=None, will output normal unfiltered data.
probability_cutoff = 0.10
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
output_dir = f"/groups/CaiLab/personal/Lex/raw/06082022_4kgenes/seqFISH_datapipeline/decoded/final_11p52_33_heg_svm_0p10_diff0_fdr10/Channel_{channel}/Pos_{JOB_ID}"

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

