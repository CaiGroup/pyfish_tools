If you are using a segmentation marker, please use the raw unaligned images for mask generation. Afterwards, run the edge deletion code. During edge deletion, the offsets generated from dapi alignment on your segmentation image will be used to shift the mask. 

The recommended workflow for generating masks using cellpose is the following:
1) Use cellpose 3 and train your own model.
2) Upload your model to the hpc and place it in the current cellpose directory.
3) Use the segment_all.batch script with the correct enviroment name and model path