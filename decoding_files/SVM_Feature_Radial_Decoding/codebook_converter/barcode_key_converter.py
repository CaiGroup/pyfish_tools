import numpy as np
import pandas as pd

def barcode_key_converter_within(codebook, ch=1):
    """This function is used to convert barcode key for within channel encodings
    
    Parameters
    -----------
    codebook = the codebook as csv
    num_hybs = number of hybs
    num_barcodes = number of barcodes
    ch = which channel this barcode key corresponds to
    
    Returns
    -------
    df = dataframe showing when the genes should appear
    """
    #get number of readout sites
    num_barcodes = len(codebook.columns)
    
    #get number of total hybs
    num_hybs = np.max(codebook.iloc[:,0])*num_barcodes
    
    #generate table containing string of zeros
    table = np.zeros(shape=(len(codebook), num_hybs)).astype(int)
    
    #calculate offset for table
    offset = int(num_hybs/num_barcodes)
    
    #get column names
    col_names = codebook.columns
    
    #fill table
    for i in range(len(codebook)):
        ps_list = []
        for j in range(num_barcodes):
            ps_list.append(int(codebook.iloc[i][col_names[j]]))
        k=0
        for ps in ps_list:
            table[i,(ps-1)+(offset*k)] = ch
            k+=1
            
    codebook_converted = pd.DataFrame(table)
    codebook_converted.index = codebook.index

    return codebook_converted

def dash_barcodekey_converter(codebook, hybcycle_code_1, hybcycle_code_2):
    """A barcode key converter for dash codebook
    
    Parameters
    ----------
    codebook = assigned pseudocolors to genes
    hybcycle_code_1 = 3 x 4 x 12 (hybcycle, colors, pseudocolor) array
                      used for barcoding round 1 and 3
    hybcycle_code_2 = 3 x 4 x 12 (hybcycle, colors, pseudocolor) array
                      used for barcoding round 2 and 4
    Returns
    -------
    df = dataframe showing when the genes should appear
    """
    
    #make indexing list to see where each pseduocolor should appear and at what channel
    code_list = []
    hyb_names = [f"Brcd{i}" for i in np.arange(1,len(codebook.columns)+1,1)]
    for _ in range(2):
        #split by barcode 1 and 2, but also use two different readout codes
        if _ < 1:
            barcode = []
            for j in range(len(codebook)):
                gene = codebook[hyb_names[0]]
                loc_1 = np.argwhere(hybcycle_code_1==gene[j])
                loc_2 = np.argwhere(hybcycle_code_2==gene[j])
                barcode.append([[loc_1[0][0],loc_1[0][1]+1],[loc_2[0][0],loc_2[0][1]+1]])
                
            code_list.append(barcode)
        else:
            barcode = []
            for j in range(len(codebook)):
                gene = codebook[hyb_names[1]]
                loc_1 = np.argwhere(hybcycle_code_1==gene[j])
                loc_2 = np.argwhere(hybcycle_code_2==gene[j])
                barcode.append([[loc_1[0][0],loc_1[0][1]+1],[loc_2[0][0],loc_2[0][1]+1]])
            code_list.append(barcode)
           
    #generate a 6 bit string where 4 positions will have 1,2,3 or 4 for channel
    decoding_barcodes = []
    for i in range(2):
        decoding = []
        decode = np.zeros(6)
        for j in range(len(code_list[0])):
            decode[code_list[i][j][0][0]]=code_list[i][j][0][1]
            decode[code_list[i][j][1][0]+3]=code_list[i][j][1][1]
            decoding.append(decode)
            decode = np.zeros(6)
        decoding_barcodes.append(decoding)
    
    #combine to 12 bit string now
    #each column will be a hyb and the numbers represent the channel
    #0 will be no spot
    hybcode_final = []
    for i in range(len(decoding_barcodes[0])):
        hybcode_final.append(np.concatenate([decoding_barcodes[0][i],decoding_barcodes[1][i]]))
        
    df = pd.DataFrame(np.array(hybcode_final).astype(int))
    
    df["Genes"] = codebook.index
    df = df.set_index("Genes")
    
    return df

def barcode_key_converter_across(codebook, num_hybs = 12, num_barcodes = 4, num_channels=4):
    """A barcode key converter for across channel cencodings
    Parameters
    ----------
    codebook = the across channel codebook
    num_hybs = number of total hybs
    num_barcodes = total number of barcodes
    num_channels = number of total channels
    
    Return
    -------
    df = dataframe showing when the genes should appear
    """
    
    #table for new codebook
    table = np.zeros(shape=(len(codebook), num_hybs)).astype(int)
    
    #generate channel table
    num_pcs = (num_hybs/num_barcodes)*num_channels
    pseudocolors = np.arange(1,num_pcs+1,1)
    channel_scheme = np.zeros(shape=((int(num_hybs/num_barcodes)),num_channels))
    k=0
    for i in range(len(channel_scheme)):
        for j in range(len(channel_scheme[0])):
            channel_scheme[i,j] = pseudocolors[k]
            k += 1
    pos_scheme = channel_scheme.astype(int)
    channel_scheme = channel_scheme.T.astype(int)
    
    #convert channel table to dictionary
    channel_dict = {}
    for i in range(len(channel_scheme)):
        for channel in channel_scheme[i]:
            channel_dict.update({channel:i+1})
            
    #make dictionary for position on table
    pos_dict = {}
    for i in range(len(pos_scheme)):
        for pos in pos_scheme[i]:
            pos_dict.update({pos:i})
    
    #fill table
    offset = int(num_hybs/num_barcodes)
    for i in range(len(codebook)):
        ps1 = codebook.iloc[i]["Hyb1"]
        ps2 = codebook.iloc[i]["Hyb2"]
        ps3 = codebook.iloc[i]["Hyb3"]
        ps4 = codebook.iloc[i]["Hyb4"]
        table[i,pos_dict[ps1]] = channel_dict[ps1]
        table[i,pos_dict[ps2]+offset] = channel_dict[ps2]
        table[i,pos_dict[ps3]+(offset*2)] = channel_dict[ps3]
        table[i,pos_dict[ps4]+(offset*3)] = channel_dict[ps4]

    codebook_converted = pd.DataFrame(table)
    codebook_converted.index = codebook["Genes"]

    return codebook_converted