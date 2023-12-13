from svm_feature_radial_decoding_v3_radcen import decode
import os
import glob
#____________________________________________________________________________________________________________________________
exp_dir                     = "231013_3k_9primers_12probes_5nM" #name of experimental directory
user                        = "Lex" #name of user
num_rounds                  = 4     #number of rounds
first_radius                = 1.25   #search radii
second_radius               = 1.5   #search radii
third_radius                = 2    #search radii
min_seed                    = 3     #how many times does a pseudocolor sequence must appear
high_exp_seed               = 3     #how many times does pseudocolor sequence must appear for highly expressed genes
total_hybs                  = 20    #number of total hybs
probability_cutoff          = 0.25  #probability cutoff for On dots (0-1). 
desired_fdr                 = 0.10  #desired FDR (0-1). Could set to None if you would like to filter yourself.
use_svm                     = True #set to false if model is not helping and to save time.
fiducial_removed            = False  #use noise removed
parity_round                = True  #do you have parity round
include_undefined           = False #do you want locations of dots that didn't pass parity
decode_high_exp_genes_first = True  #do you want to decode highly expressed genes first
triple_decode               = False  #do you want to perform an additional third round of decoding
score_brightness            = True  #do you want to use brightness for scoring 
blank_ch                    = False #will you have a blank channel
#____________________________________________________________________________________________________________________________
#No need to really adjust unless you want alternative input or output paths
JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0) 
if fiducial_removed == True:
    locations_path = glob.glob(f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/dots_detected/Channel_All/Pos{JOB_ID}/noise_removed*")
else:
    locations_path = glob.glob(f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/dots_detected_radial_centered/Channel_All/Pos{JOB_ID}/locations*")
#general codebook path
codebook_path = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/barcode_key/codebook_string_across.csv"
#how many allowed drops in calls 
if parity_round == True:
    for diff in range(2):
        #Where do you want to output the files
        output_dir = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/decoded_v3_rad_heg/final_{first_radius}{second_radius}{third_radius}_seed{min_seed}{high_exp_seed}_hegoff_svm_p{probability_cutoff*100}_diff{diff}_fdr{desired_fdr*100}/Channel_All/Pos_{JOB_ID}"
        if len(locations_path) > 1:
            for locations in locations_path:
                decoder = decode(location_path=locations, codebook_path=codebook_path,
                                 num_rounds=num_rounds, first_radius=first_radius, 
                                 second_radius=second_radius,third_radius=third_radius,
                                 diff=diff, min_seed=min_seed, high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                                 probability_cutoff=probability_cutoff, desired_fdr = desired_fdr, 
                                 output_dir=output_dir, include_undefined=include_undefined, 
                                 decode_high_exp_genes_first=decode_high_exp_genes_first,
                                 triple_decode=triple_decode, parity_round=parity_round, 
                                 score_brightness = score_brightness, blank_ch=blank_ch)
                decoder.feature_radial_decoding(use_svm=use_svm)
        else:
            decoder = decode(location_path=locations_path[0], codebook_path=codebook_path,
                             num_rounds=num_rounds, first_radius=first_radius,
                             second_radius=second_radius,
                             third_radius=third_radius, diff=diff, min_seed=min_seed,
                             high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                             probability_cutoff=probability_cutoff,desired_fdr = desired_fdr,
                             output_dir=output_dir, include_undefined=include_undefined,
                             decode_high_exp_genes_first=decode_high_exp_genes_first,
                             triple_decode=triple_decode, parity_round=parity_round, 
                             score_brightness = score_brightness, blank_ch=blank_ch)
            decoder.feature_radial_decoding(use_svm=use_svm)
else:
    #Where do you want to output the files
    output_dir = f"/groups/CaiLab/personal/{user}/raw/{exp_dir}/pyfish_tools/output/decoded_v3/final_{first_radius}{second_radius}{third_radius}_seed{min_seed}{high_exp_seed}_heg_svm_p{probability_cutoff*100}_diff0_fdr{desired_fdr*100}/Channel_All/Pos_{JOB_ID}"
    if len(locations_path) > 1:
        for locations in locations_path:
            decoder = decode(location_path=locations, codebook_path=codebook_path,
                             num_rounds=num_rounds, first_radius=first_radius, 
                             second_radius=second_radius,third_radius=third_radius,
                             diff=0, min_seed=min_seed, high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                             probability_cutoff=probability_cutoff, desired_fdr = desired_fdr, 
                             output_dir=output_dir, include_undefined=include_undefined, 
                             decode_high_exp_genes_first=decode_high_exp_genes_first,
                             triple_decode=triple_decode, parity_round=parity_round, 
                             score_brightness = score_brightness, blank_ch=blank_ch)
            decoder.feature_radial_decoding(use_svm=use_svm)
    else:
        decoder = decode(location_path=locations_path[0], codebook_path=codebook_path,
                         num_rounds=num_rounds, first_radius=first_radius,
                         second_radius=second_radius,
                         third_radius=third_radius, diff=0, min_seed=min_seed,
                         high_exp_seed=high_exp_seed, total_hybs=total_hybs, 
                         probability_cutoff=probability_cutoff,desired_fdr = desired_fdr,
                         output_dir=output_dir, include_undefined=include_undefined,
                         decode_high_exp_genes_first=decode_high_exp_genes_first,
                         triple_decode=triple_decode, parity_round=parity_round, 
                         score_brightness = score_brightness, blank_ch=blank_ch)
        decoder.feature_radial_decoding(use_svm=use_svm)
