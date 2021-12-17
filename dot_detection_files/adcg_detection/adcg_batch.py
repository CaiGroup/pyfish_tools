#data management
from pathlib import Path
import os
from webfish_tools.util import find_matching_files
import re
import numpy as np
import tifffile as tf
import pandas as pd
#importing julia
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main

#load adch function
jl.eval('include("adcg_function.jl")')

#get JOB ID
JOB_ID = os.getenv('SLURM_ARRAY_TASK_ID', 0)
print(f'This is task {JOB_ID}')

#get path to pre processed images
directory = Path("/groups/CaiLab/personal/Lex/raw/Linus_10k_cleared_080918_NIH3T3/notebook_pyfiles/pre_processed_images")
position_name = 'MMStack_Pos{pos}.tif'
files, _, _ = find_matching_files(directory, f'HybCycle_{JOB_ID}' + f'/{position_name}')
files = [str(f) for f in files]

#gen dir
parent_name = Path(files[0]).parent.parent.parent

#define channels (all or specific channel)
channel = 0

#loop through each path
for path in files:
    #read in tiff
    data = tf.imread(path)
    #make output dir
    hybcycle = Path(path).parent.name
    image_name = Path(path).name
    output_dir = parent_name/"dots_detected"/"ADCG"/hybcycle
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(data.shape)==3:
        if channel == "all":
            #loop through channels
            for c in range(data.shape[0]):
                output_path = output_dir/image_name
                Main.data = data[c]
                Main.path = str(output_path).replace(".tif",f"_channel_{c+1}.csv")
                Main.channel = c+1
                jl.eval("adcg(data, path, channel)")
        else:
            output_path = output_dir/image_name
            Main.data = data[channel]
            Main.path = str(output_path).replace(".tif",f"_channel_{channel+1}.csv")
            Main.channel = channel+1
            jl.eval("adcg(data, path, channel)")
    else:
        if channel == "all":
            #loop through z
            for z in range(data.shape[0]):
                #loop through channels
                for c in range(data.shape[1]-1):
                    output_path = output_dir/image_name
                    Main.data = data[z][c]
                    Main.path = str(output_path).replace(".tif",f"_zslice_{z}_channel_{c+1}.csv")
                    Main.channel = c+1
                    jl.eval("adcg(data, path, channel)")
        else:
            #loop through z
            for z in range(data.shape[0]):
                output_path = output_dir/image_name
                Main.data = data[z][channel]
                Main.path = str(output_path).replace(".tif",f"_zslice_{z}_channel_{channel+1}.csv")
                Main.channel = channel+1
                jl.eval("adcg(data, path, channel)")
