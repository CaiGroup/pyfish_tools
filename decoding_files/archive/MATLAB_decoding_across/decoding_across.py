from datapipeline.decoding.decoding_class import Decoding
import tifffile as tf
import numpy as np
import os

#get MATLAB running
PATH = os.getenv('PATH')
os.putenv('PATH', ':'.join(['/software/Matlab/R2019a/bin', PATH]))

def decode_across(data_dir,position,decoded_dir, locations_dir, position_dir, barcode_dst, barcode_src, synd_decoding=False, lvf= None,zvf=None,lwvf = None):
    
#     mask = tf.imread(labeled_src)
#     new_mask = []
    
#     if propagate_mask == True:
#         for _ in range(z):
#             new_mask.append(mask)
    
#     new_mask = np.array(new_mask)
    
    
    decoder = Decoding(data_dir = data_dir, 
                        position = position, 
                        decoded_dir = decoded_dir, 
                        locations_dir = locations_dir, 
                        position_dir = position_dir, 
                        barcode_dst = barcode_dst, 
                        barcode_src = barcode_src , 
                        bool_decoding_with_previous_dots = False, 
                        bool_decoding_with_previous_locations = False, 
                        bool_fake_barcodes = True, 
                        bool_decoding_individual = [1], 
                        min_seeds = 'number_of_rounds - 1', 
                        allowed_diff = 1, 
                        dimensions = 3, 
                        num_zslices = 2, 
                        segmentation = False, 
                        decode_only_cells = False, 
                        labeled_img = None, 
                        num_wav = 3, 
                        synd_decoding = synd_decoding,
                        Hpath=None,
                        lvf= lvf,
                        zvf=zvf,
                        lwvf = lwvf,
                        lampfish_pixel=False,
                        start_time = 0
                  )
    
    decoder.run_decoding_across_2d()