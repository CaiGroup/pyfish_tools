import pandas as pd
import numpy as np
from pathlib import Path

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
            #get threshold number
            Threshold = Path(gene_loc_dir[i]).parent.name
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
                counts = pd.DataFrame(cell.pivot_table(columns=["genes"], aggfunc='size'), columns=[f"cell{j}"])
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
            #read in df
            pos = pd.read_csv(gene_loc_dir[i], index_col=0)
            #get counts of each gene per cell
            cell_counts = []
            for j in np.unique(pos["cell number"].values):
                cell = pos[pos["cell number"]==j]
                counts = pd.DataFrame(cell.pivot_table(columns=["genes"], aggfunc='size'), columns=[f"cell{j}_pos{i}"])
                cell_counts.append(counts)
            #combine 
            genebycell = pd.concat(cell_counts, axis=1)  
            genebycell = genebycell.fillna(value=0)
            genebycell_list.append(genebycell)
        
        #combine all dfs
        final_genebycell = pd.concat(genebycell_list, axis=1) 
        final_genebycell = final_genebycell.fillna(value=0)
        
        #write file
        final_genebycell.to_csv(str(final_output))