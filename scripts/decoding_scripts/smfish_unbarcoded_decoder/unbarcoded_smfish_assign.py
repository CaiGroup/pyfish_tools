"""
author: Katsuya Lex Colon
group: Cai Lab
date:05/01/2022
"""

#import gen packages
import pandas as pd
from pathlib import Path
#parallel processing
from concurrent.futures import ProcessPoolExecutor

def assign_genes_to_spots(location_path, codebook_path, output_dir=None):
    """
    A function to assign gene names to unbarcoded smFISH spots.
    
    Parameters
    ----------
    locations: dot detected locations file
    codebook: unbarcoded codebook (genes, hybcycle, channel)
    output_dir: path to desired output directory
    """
    
    #read in locations file and drop unnamed columns
    locations = pd.read_csv(location_path)
    unwanted_columns = []
    for names in locations.columns:
        if "Unnamed" in names:
            unwanted_columns.append(names)
    locations = locations.drop(unwanted_columns, axis=1)
    
    #read in codebook
    codebook = pd.read_csv(codebook_path)
    
    #collect z info
    location_path_name = Path(location_path).name
    z_info = location_path_name.split("_")[2].replace(".csv","")
    
    #make directories
    Path(output_dir).mkdir(parents=True, exist_ok = True)
    output_path = Path(output_dir) / f"assigned_smfish_z_{z_info}.csv"

    #convert hyb and channel into a list of tuples from dot locations csv
    codes = locations[["hyb","ch"]].values.astype(int)
    tuple_codes = [tuple(i) for i in codes]
    
    #create keys and values from codebook
    values = codebook[["HybCycle","Channel"]].values
    keys = codebook["Genes"].values

    #create dictionary for assignment
    smfish_dict = {}
    for i in range(len(keys)):
        smfish_dict.update({tuple(values[i]):keys[i]})
     
    #convert tuple codes from locations to gene names using dictionary
    gene_names = []
    for code in tuple_codes:
        try:
            gene_names.append(smfish_dict[code])
        except KeyError:
            gene_names.append("Undefined")
        
    #replace hyb and channel info with gene name
    assigned_loc = locations.drop(["hyb","ch"], axis=1)
    assigned_loc.insert(loc=0, column='genes', value=gene_names)
    
    #write out files
    assigned_loc.to_csv(str(output_path), index=False)
    
def assign_genes_to_spots_parallel(locations, codebook, output_dir):
    
    """
    Runs assignment in parallel fashion.
    
    Parameters
    ----------
    locations: list of spot locations file path
    codebook: single codebook path
    output_dir: desired output directory
    """
    
    with ProcessPoolExecutor(max_workers=12) as exe:
        for location in locations:
            exe.submit(assign_genes_to_spots, location_path=location, codebook_path=codebook, output_dir=output_dir)
    
