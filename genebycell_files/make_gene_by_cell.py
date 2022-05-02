"""
author: Katsuya Lex Colon
group: Cai Lab
updated: 01/07/22
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def make_genebycell(gene_loc_dir, output_dir = None, check_thresholds=True, channel = "all"):
    """
    This function will generate a gene by cell matrix.
    Parameters
    ----------
    gene_loc_dir: gene locations directory
    output_dir: output path
    check_thresholds: bool to make gene by cell on various thresholds tested
    channel: 1,2,3,4 or all 
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
        for i in range(len(gene_loc_dir)):
            try:
                #read in df
                pos = pd.read_csv(gene_loc_dir[i], index_col=0)
            except FileNotFoundError:
                #output readme file if file not found
                path = final_output.parent / "missing_files.txt"
                #check if text file exists
                if os.path.isfile(str(path)):
                    with open(str(path),"a") as f:
                        f.write(f"{str(Path(gene_loc_dir[i]).parent.name)} gene locations missing. Could be that either no genes were decoded in this position or no cells are present (check image or masks)." + "\n")
                        f.close()
                    continue
                else:
                    with open(str(path),"w+") as f:
                        f.write(f"{str(Path(gene_loc_dir[i]).parent.name)} gene locations missing. Could be that either no genes were decoded in this position or no cells are present (check image or masks)." + "\n")
                        f.close()
                    continue
            #get counts of each gene per cell
            cell_counts = []
            for j in np.unique(pos["cell number"].values):
                cell = pos[pos["cell number"]==j]
                #check cell number if int, if not then drop since it maye have grabbed spots from other cells
                if (float(j)).is_integer():
                    counts = pd.DataFrame(cell.pivot_table(columns=["genes"], aggfunc='size'), columns=[f"cell{j}_pos{i}"])
                else:
                    continue
                cell_counts.append(counts)
            #combine 
            try:
                genebycell = pd.concat(cell_counts, axis=1)  
            except ValueError:
                continue  
            genebycell = genebycell.fillna(value=0)
            genebycell_list.append(genebycell)
        
        #combine all dfs
        final_genebycell = pd.concat(genebycell_list, axis=1) 
        final_genebycell = final_genebycell.fillna(value=0)
        
        #write file
        final_genebycell.to_csv(str(final_output))
