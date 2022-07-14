"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 07/14/22
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tifffile as tf
from skimage.measure import regionprops
import os

def calc_density(locations, mask, pos=0, counts_threshold = 100, pixel=0.11):
    
    """
    This function will compute the gene density per cell.
    
    Parameters
    ----------
    locations: gene locations file after decoding
    mask: cell mask array
    pos: position number
    counts_threshold: total counts each cell must be above
    pixel: pixel size in micrometer
    """
    
    #get area per cell
    area_per_cell = []
    info = regionprops(mask)
    for cell in info:
        area = cell.area
        label = cell.label
        area_per_cell.append([label,area])
    df = pd.DataFrame(area_per_cell)
    df.columns = ["cell number","area"]
    
    #get gene density per cell
    gene_density = []
    for cell in locations["cell number"].unique():
        #check cell number if int, if not then drop since it maye have grabbed spots from other cells
        if (float(cell)).is_integer():
            gene_counts = locations[locations["cell number"]==cell].groupby("genes").size()
            if gene_counts.sum() < counts_threshold:
                continue
            gene_counts = gene_counts / df[df["cell number"] == cell].area.iloc[0]
            gene_counts = gene_counts/(pixel**2)
            final_counts = pd.DataFrame(gene_counts)
            final_counts.columns = [f"cell{cell}_pos{pos}"] 
            gene_density.append(final_counts)
        else:
            continue
            
    gene_density = pd.concat(gene_density, axis=1)  
        
    return gene_density

def make_genebycell(gene_loc_dir, mask_dir=None, output_dir = None,  
                    counts_threshold = 100, pixel = 0.11,
                    channel = 1, check_thresholds=False):
    """
    This function will generate a counts gene by cell matrix and cell size normalized gene density matrix.
    Parameters
    ----------
    gene_loc_dir: list of gene locations
    mask_dir: path to mask directory
    output_dir: output path
    counts_threshold: total counts each cell must be above
    pixel: pixel length 
    channel: 1,2,3,4 or all 
    check_thresholds: bool to make gene by cell on various thresholds tested
    """
    if check_thresholds == True:
        #go through each threshold csv
        for i in range(len(gene_loc_dir)):
            Threshold = ""
            parent_dir = Path(gene_loc_dir[i]).parent
            #get threshold number
            while Threshold.find("Threshold") == -1:
                Threshold = parent_dir.name
                parent_dir = parent_dir.parent
            #across all channels or not
            if channel == "all":
                final_output = Path(output_dir) / Threshold / "genebycell.csv"
            else:
                final_output = Path(output_dir) / Threshold / f"genebycell_{channel}.csv"
            #read in df
            pos = pd.read_csv(gene_loc_dir[i], index_col=0)
            #get counts of each gene per cell
            cell_counts = []
            for j in np.unique(pos["cell number"].values):
                cell = pos[pos["cell number"]==j]
                #check cell number if int, if not then drop since it maye have grabbed spots from other cells
                if (float(j)).is_integer():
                    counts = pd.DataFrame(cell.pivot_table(columns=["genes"], aggfunc='size'), columns=[f"cell{j}"])
                else:
                    continue
                cell_counts.append(counts)
            #combine 
            genebycell = pd.concat(cell_counts, axis=1)  
            genebycell = genebycell.fillna(value=0)
            #make directory
            final_output.parent.mkdir(parents=True, exist_ok=True)
            #write file
            genebycell.to_csv(str(final_output))
    else:
        #across all channels or not
        if channel == "all":
            final_output = Path(output_dir) / "final" / "genebycell.csv"
        else:
            final_output = Path(output_dir) / "final" / f"genebycell_{channel}.csv"
            
        #make directory
        final_output.parent.mkdir(parents=True, exist_ok=True)
        
        #all positions
        genebycell_list = []
        gene_den_list = []
        for i in range(len(gene_loc_dir)):
            try:
                #read in df
                pos = pd.read_csv(gene_loc_dir[i], index_col=0)
                #check if df is empty
                if len(pos)==0:
                    raise FileNotFoundError()
            except FileNotFoundError:
                #output readme file if file not found
                path = final_output.parent / "missing_files.txt"
                #check if text file exists
                if os.path.isfile(str(path)):
                    with open(str(path),"a") as f:
                        f.write(f"Channel {channel}, {str(Path(gene_loc_dir[i]).parent.name)} gene locations missing. Could be that either no genes were decoded in this position, FDR cutoff was too stringent, or no cells are present (check image or masks)." + "\n")
                        f.close()
                    continue
                else:
                    with open(str(path),"w+") as f:
                        f.write(f"Channel {channel}, {str(Path(gene_loc_dir[i]).parent.name)} gene locations missing. Could be that either no genes were decoded in this position, FDR cutoff was too stringent, or no cells are present (check image or masks)." + "\n")
                        f.close()
                    continue
            #get counts of each gene per cell
            cell_counts = []
            #get position name
            pos_name = Path(gene_loc_dir[i]).parent.name
            for j in np.unique(pos["cell number"].values):
                cell = pos[pos["cell number"]==j]
                #check cell number if int, if not then drop since it maye have grabbed spots from other cells
                if (float(j)).is_integer():
                    counts = pd.DataFrame(cell.pivot_table(columns=["genes"], aggfunc='size'), columns=[f"Cell{j}_{pos_name}"])
                else:
                    continue
                #check if cell meets counts criteria
                if counts.sum().iloc[0] < counts_threshold:
                    continue
                else:
                    cell_counts.append(counts)
            #combine 
            try:
                genebycell = pd.concat(cell_counts, axis=1)  
            except ValueError:
                continue  
            genebycell = genebycell.fillna(value=0)
            genebycell_list.append(genebycell)
            
            #generate gene density matrix
            if mask_dir != None:
                #calculate density per z
                z_density = []
                for z in pos["z"].unique():
                    pos_z = pos[pos["z"]==z].reset_index(drop=True)
                    try:
                        mask_path = Path(mask_dir) / f"MMStack_{pos_name.replace('_','')}_z{int(z)}.tif"
                        mask = tf.imread(str(mask_path))
                    except FileNotFoundError:
                        mask_path = Path(mask_dir) / f"MMStack_{pos_name.replace('_','')}.tif"
                        mask = tf.imread(str(mask_path))
                    pos_number = int(pos_name.split("_")[1])
                    gene_density_mtx = calc_density(pos_z, mask, pos=pos_number, 
                                                    counts_threshold = counts_threshold, pixel = pixel)
                    z_density.append(gene_density_mtx)
                #get average density across z for same gene
                if len(z_density) == 1:
                    gene_den_list.append(z_density[0])
                else:
                    comb = pd.concat(z_density)
                    comb = comb.groupby(comb.index).mean()
                    final = comb.fillna(0)
                    gene_den_list.append(final)

        #combine all dfs
        final_genebycell = pd.concat(genebycell_list, axis=1) 
        final_genebycell = final_genebycell.fillna(value=0)
        
        #combine all gene density mtx
        if mask_dir != None:
            gene_den_df = pd.concat(gene_den_list,axis=1).fillna(0)
        
        #write file
        final_genebycell.to_csv(str(final_output))
        gene_den_df.to_csv(str(final_output.parent / f"gene_density_{channel}.csv"))
