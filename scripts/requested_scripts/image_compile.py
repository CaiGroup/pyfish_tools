import pandas as pd
import tifffile as tf
import os 
import numpy as np
import re

def sort_files(filename):
    # pattern that matches two sets of one or more digits (\d+) separated by '_'
    # followed by '.tif'
    match = re.search(r'(\d+)_(\d+)\.tif', filename)
    if match:
        # if there is a match, convert both groups to int and return as tuple
        return (int(match.group(1)), int(match.group(2)))
    else:
        # if there's no match, return 0 (or whatever default value makes sense)
        return (0, 0)

def create_stack(path, channel_names = ["640", "561", "488", "405"]):
    # grab position directories from provided path
    if path[-1] == "/":
        path = path[:-1]
    wd = os.listdir(path)
    # only keep directories with "Pos" in folder name
    pos_dirs = [folder for folder in wd if "Pos" in folder]
    for pos in pos_dirs:
        #convert to array
        images = np.array(os.listdir(pos))
        #sort the image names by z and channel
        sorted_files = sorted(images, key=sort_files)
        #remove any files that is not tif
        sorted_files = [tif for tif in sorted_files if ".tif" in tif]
        #now we will read each file, compile and write out as stack
        #first lets group channels
        separated = []
        for name in channel_names:
            separated.append([file for file in sorted_files if name in file])
        #now we will zip them so that the same z's are grouped together
        decorator = "separated[0]"
        for i in range(1, len(channel_names),1):
            decorator += f",separated[{i}]"
        func = "zip(" + decorator + ")"
        evaluate = list(eval(func))
        #now we will read them
        arr_stacks = []
        for stack in evaluate:
            z_stack_read = []
            for image in stack:
                full_path = path + f"/{pos}" + f"/{image}"
                img = tf.imread(full_path)
                z_stack_read.append(img)
            arr_stacks.append(np.array(z_stack_read))
        #now we will write it
        pos_num = pos.replace("Pos", "")
        tf.imwrite(path+f"/MMStack_Pos{pos_num}.ome.tif", np.array(arr_stacks))
        print(path+f"/MMStack_Pos{pos_num}.ome.tif")

num_cycles = int(input("How Many HybCycles?:"))
path = input("Path to Experimental Directory (don't include slash at the end):")
channel_num = int(input("How many channels?:"))
channel_names = ["640", "561", "488", "405"]
channels = channel_names[:channel_num]
for cycle in range(0,num_cycles+1,1):
    path = path + f"/HybCycle_{cycle}"
    create_stack(path, channels)