'''
Author: Arun Chakravorty
Purpose:  Functions used to generate a codebook with equalized FPKMS across all hybs based on known expression data. Note that these functions are for 5 barcoding rounds specifically.

Algorithm works by assigning highest expressing genes to orthogonal barcodes. Subsequently, a search is performed to find a barcode that encompasses hybs with lower expression values relative to the other hybs. Iteratively, the next highest expressing gene is asssigned to the found barcode. If a barcode can not be found (usually once the codebook is close to full), a random barcode from those remaining is assigned.
'''

import pandas as pd
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
import plotly.express as px
import random

def getFPKMS(channel, maxlist):
    '''
    Inputs:
    channel - pandas dataframe containing genes desired under column heading 'ensemble id'
    maxlist - pandas dataframe containing list of genes, under 'gene_symbols', and their associated expression, under 'avg' 

    Returns a pandas dataframe containing genes desired, under 'gene_symbols', and relative expression, under 'avg'. 
    '''
    Channel_FPKM = maxlist[maxlist['gene_symbols'].isin(channel['ensemble id'])]
    Channel_FPKM.reset_index(drop = True)
    
    Channel_notfound = channel[~channel['ensemble id'].isin(maxlist['gene_symbols'])]
    
    NotFound = pd.DataFrame(Channel_notfound['ensemble id'])
    NotFound['avg'] = 0
    NotFound = NotFound.rename(columns={'ensemble id': 'gene_symbols'})
    
    Finaldf = pd.concat([Channel_FPKM, NotFound])
    Finaldf = Finaldf.sort_values(by = 'avg', ascending = False)
    
    Finaldf = Finaldf.reset_index(drop = True)
    return Finaldf



def AssignHighExpressionCodewords(FPKM_sorted, pseudocolors):
    '''
    Inputs:
    FPKM_sorted - Genes under 'gene_symbols', and expression values under 'avg'. Must be sorted from highest to lowest expression values. 
    pseudocolors - Integer value for number of pseudocolors

    Returns a pandas dataframe containing fully orthogonal barcodes for the top n number of genes, where n is the number of pseudocolors.
    '''
    HighExpressionCodewords = []

    Range = []
    
    for n in np.arange(1.0, pseudocolors + 1, 1.0):
        Range.append(n)

    
    for i in range(0,pseudocolors):
        hyb1 = Range[i]
        hyb2 = Range[(i+1)%pseudocolors]
        hyb3 = Range[(i+2)%pseudocolors]
        hyb4 = Range[(i+3)%pseudocolors]
        hyb5 = Range[(int(hyb1 + hyb2 + hyb3 + hyb4)%pseudocolors-1)]

        HighExpressionCodewords.append([hyb1, hyb2, hyb3, hyb4, hyb5])

    HighExpressionCodewords


    Codewords = pd.DataFrame(HighExpressionCodewords, columns = ['hyb1', 'hyb2', 'hyb3', 'hyb4', 'hyb5'])
    Codewords

    
    gene = ['Gene{}'.format(i) for i in range(1,(Codewords .shape[0] + 1))]
    Codewords ['Gene'] = gene


    FPKM_sorted = FPKM_sorted.reset_index(drop = True)
    
    for i in range(len(Codewords)):
        Codewords['Gene'] = FPKM_sorted['gene_symbols'][0:len(Codewords)]
        
    return Codewords


def FindRemainingCodes(UsedCodebook, EntireCodebook):
    '''
    Inputs: 
    UsedCodebook - Codewords already assigned to genes. Contains gene name under 'Gene' and the pseudcolor value assigned under 'hyb1', 'hyb2', etc. 
    EntireCodebook - Pandas dataframe containing every possible Codeword. 
    
    Returns a pandas dataframe containing the remaining Codewords available.
    '''
    
    RemainingCodebook = EntireCodebook.copy()
    
    for i in range(0, len(UsedCodebook)):
        
        RemainingCodebook = RemainingCodebook.drop(RemainingCodebook[(RemainingCodebook['hyb1'] == UsedCodebook['hyb1'][i])
                & (RemainingCodebook['hyb2'] == UsedCodebook['hyb2'][i])
                & (RemainingCodebook['hyb3'] == UsedCodebook['hyb3'][i])
                & (RemainingCodebook['hyb4'] == UsedCodebook['hyb4'][i])
                & (RemainingCodebook['hyb5'] == UsedCodebook['hyb5'][i])
                
                 ].index)
                           
    RemainingCodebook = RemainingCodebook.reset_index(drop = True)
    
    return RemainingCodebook



def FPKMSforCodebook(Codebook, FPKMS_table, numColors):
    '''
    Inputs: 
    Codebook - Pandas dataframe with gene name, under 'Gene', and codeword assigninment under 'hyb1', 'hyb2' etc.
    FPKMS_table - Pandas dataframe containing the genes under 'gene_symbols', and the expression value under 'avg'.
    numColors - Integer number of pseudocolors in codebook design.
    
    Returns a pandas dataframe containing the average expression for each hybridization. Columns are each barcoding round, while each row represents the pseudocolor. 
    '''
    
    counts_per_hyb= pd.DataFrame(0., index = np.arange(numColors+1) ,columns = ['hyb1', 'hyb2,', 'hyb3', 'hyb4', 'hyb5'])
    counts_per_hyb.columns = ['hyb1', 'hyb2', 'hyb3', 'hyb4', 'hyb5']
    
    genelist = Codebook['Gene']
    
    for gene in genelist:
        #For each gene we create a localsubset from the df_probe_coded
        localsubset = Codebook.loc[Codebook['Gene'] == gene]
        localsubset = localsubset.reset_index(drop = True)
        #Stores the pseudocolor each hyb for this specific gene
        hyb1_pc = localsubset['hyb1'][0]
        hyb2_pc = localsubset['hyb2'][0]
        hyb3_pc = localsubset['hyb3'][0]
        hyb4_pc = localsubset['hyb4'][0]
        hyb5_pc = localsubset['hyb5'][0]
        # We find the fpkm for this specific gene now and add it to the correct dataframe 
        fpkm = FPKMS_table.loc[FPKMS_table['gene_symbols'] == gene]['avg'].values
        counts_per_hyb['hyb1'][hyb1_pc]+= fpkm
        counts_per_hyb['hyb2'][hyb2_pc]+= fpkm
        counts_per_hyb['hyb3'][hyb3_pc]+= fpkm
        counts_per_hyb['hyb4'][hyb4_pc]+= fpkm
        counts_per_hyb['hyb5'][hyb5_pc]+= fpkm
        
    return counts_per_hyb.iloc[1: , :]


def FindNextCodeWord(FPKMS, pseudocolors):
    '''
    Algorithmically, this functions generates a potential barcode by determining which hybridizations have less expression relative to the other hybdrizations within each barcoding round. Incorporates a level of random shuffling to select a usable barcode among a number potential barcodes.
    
    Inputs:
    FPKMS - Pandas dataframe containing the genes under 'gene_symbols', and the expression value under 'avg'.
    pseudocolors - Integer number of pseudocolors in codebook design.
    
    Returns an array for a barcode for the next gene that would help equalize expression for across hybridizations. 
    '''

    OrderOfLeast = [1,2,3,4,5]
    random.shuffle(OrderOfLeast)

    #When we shuffle this list: 
    # 3 - will be the pseudocolor that is calculate depending on the others

    #print(OrderOfLeast)

    if (OrderOfLeast[0] != 5):
        Hyb1 = int(FPKMS['hyb1'].nsmallest(OrderOfLeast[0]).index[OrderOfLeast[0]-1])
        #print('Hyb1', Hyb1)

    if (OrderOfLeast[1] != 5):
        Hyb2 = int(FPKMS['hyb2'].nsmallest(OrderOfLeast[1]).index[OrderOfLeast[1]-1])
        #print('Hyb2', Hyb2)

    if (OrderOfLeast[2] != 5):
        Hyb3 = int(FPKMS['hyb3'].nsmallest(OrderOfLeast[2]).index[OrderOfLeast[2]-1])
        #print('Hyb3', Hyb3)

    if (OrderOfLeast[3] != 5):
        Hyb4 = int(FPKMS['hyb4'].nsmallest(OrderOfLeast[3]).index[OrderOfLeast[3]-1])
        #print('Hyb4', Hyb4)
        
    if (OrderOfLeast[4] != 5):
        Hyb5 = int(FPKMS['hyb5'].nsmallest(OrderOfLeast[4]).index[OrderOfLeast[4]-1])
        #print('Hyb5', Hyb5)

    ParityHyb = OrderOfLeast.index(5) + 1
    #print(ParityHyb)

    #Range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Range = []
    for n in np.arange(1, pseudocolors + 1, 1):
        Range.append(n)

    if (ParityHyb == 1):
        Hyb1 = int(InverseModulo(Hyb2, Hyb3, Hyb4, Hyb5, pseudocolors))
        #print('Hyb1', Hyb1)

    if (ParityHyb == 2):
        Hyb2 = int(InverseModulo(Hyb1, Hyb3, Hyb4, Hyb5, pseudocolors))
        #print('Hyb2', Hyb2)

    if (ParityHyb == 3):
        Hyb3 = int(InverseModulo(Hyb1, Hyb2, Hyb4, Hyb5, pseudocolors))
        #print('Hyb3', Hyb3)

    if (ParityHyb == 4):
        Hyb4 = int(InverseModulo(Hyb1, Hyb2, Hyb3, Hyb5, pseudocolors))
        #print('Hyb4', Hyb4)
        
    if (ParityHyb == 5):
        Hyb5 = int(Range[int(Hyb1 + Hyb2 + Hyb3 + Hyb4)%pseudocolors-1])
        #print('Hyb4', Hyb4)
    
    return [Hyb1, Hyb2, Hyb3, Hyb4, Hyb5]


def InverseModulo(FirstHyb, SecondHyb, ThirdHyb, ParityValue, pseudocolors):
    '''
    An inverse modulus function to caluculate a barcoding round pseudocolor assuming Parity Check, where (hyb1 + hyb2 + hyb3 + hyb4)%pseudcolors = hyb5.
    
    Returns an integer.
    '''
    PossibleValues = []
    for n in np.arange(1.0, pseudocolors + 1, 1.0):
        PossibleValues.append(n)
    
    for i in range(0,pseudocolors): 
        if PossibleValues[(FirstHyb + SecondHyb + ThirdHyb + i)%pseudocolors] == ParityValue:
            return PossibleValues[i] 
        else: 
            i += 1

            
def CheckIfCodeIsPresent(codeword, RemainingCodebook):
    '''
    codeword - array containing barcode
    RemainingCodebook - pandas dataframe containing the remaining Codewords available.
    
    Returns True if codeword is present in RemainingCodebook, False if not. 
    '''
    tempCodebook = RemainingCodebook.drop(labels = ['Gene'], axis = 1)
    tempNumpy = tempCodebook.to_numpy()
    
    for a in tempNumpy:
        if np.array_equal(codeword, a):
            return(True)

    return(False)


def CreateFullCodebook(UsedCodebook, EntireCodebook, FPKMS_table, numColors):
    '''
    Creates equalized codebook based on known gene and expression data. 
    
    UsedCodebook - Pandas dataframe containing genes and barcodes already assigned. 
    EntireCodebook - Pandas dataframe containing all possible barcodes 
    FPKMS_table - Pandas dataframe containing the genes under 'gene_symbols', and the expression value under 'avg'.
    numColors - Integer number of pseudocolors in codebook design.

    
    Returns a pandas dataframe containing all genes and their barcode assignment.
    '''
    
    FPKMS_table = FPKMS_table.reset_index(drop = True)  
    counter = 0
    
    FPKMS = FPKMSforCodebook(UsedCodebook, FPKMS_table, numColors)
    RemainingCodebook = FindRemainingCodes(UsedCodebook, EntireCodebook)
    
    while len(UsedCodebook)<len(FPKMS_table):
    
        
        #RemainingCodebook = FindRemainingCodes(UsedCodebook, EntireCodebook)
        #FPKMS = FPKMSforCodebook(UsedCodebook, FPKMS_table, numColors)
        
        print('RemainingCodebook Length', len(RemainingCodebook))

        ## Now we want to identify the lowest 1st hyb, 2nd lowest 2nd hyb....

        NextCode = FindNextCodeWord(FPKMS, numColors)
        print(NextCode)
        #Check if the code is in the RemainingCodebook?
        i = 0

        while (i<50): 
            if (CheckIfCodeIsPresent(NextCode, RemainingCodebook)):
                NumGenesInCodebook = len(UsedCodebook)
                NextGene = FPKMS_table['gene_symbols'][NumGenesInCodebook]
                FPKM_toadd = FPKMS_table['avg'][NumGenesInCodebook]
                
                ArrayToAppend = NextCode + [NextGene]
                TempDfToAppend = pd.DataFrame(columns = ['hyb1', 'hyb2', 'hyb3', 'hyb4', 'hyb5','Gene'])
                TempDfToAppend = TempDfToAppend.append({
                    'hyb1': ArrayToAppend[0],
                    'hyb2': ArrayToAppend[1],
                    'hyb3': ArrayToAppend[2],
                    'hyb4': ArrayToAppend[3],
                    'hyb5': ArrayToAppend[4],
                    'Gene': ArrayToAppend[5]
                }, ignore_index = True)
                UsedCodebook = pd.concat((UsedCodebook, TempDfToAppend))
                UsedCodebook = UsedCodebook.reset_index(drop = True)
                
                
                assert(len(UsedCodebook) == NumGenesInCodebook+1)
                #print(len(UsedCodebook))
                #print(NumGenesInCodebook)
                
                FPKMS['hyb1'][ArrayToAppend[0]]+= FPKM_toadd
                FPKMS['hyb2'][ArrayToAppend[1]]+= FPKM_toadd
                FPKMS['hyb3'][ArrayToAppend[2]]+= FPKM_toadd
                FPKMS['hyb4'][ArrayToAppend[3]]+= FPKM_toadd
                FPKMS['hyb5'][ArrayToAppend[4]]+= FPKM_toadd
                
                #Remove the code from RemainingCodebook
                RemainingCodebook = RemainingCodebook.drop(RemainingCodebook[(RemainingCodebook['hyb1'] == ArrayToAppend[0])
                    & (RemainingCodebook['hyb2'] == ArrayToAppend[1])
                    & (RemainingCodebook['hyb3'] == ArrayToAppend[2])
                    & (RemainingCodebook['hyb4'] == ArrayToAppend[3])
                    & (RemainingCodebook['hyb5'] == ArrayToAppend[4])].index)
                           
                RemainingCodebook = RemainingCodebook.reset_index(drop = True)
                
                #Set i to a high number so that it exits from the for loop and also doesn't activate random assignment
                i = 269 

            else: 
                NextCode = FindNextCodeWord(FPKMS, numColors)
                i += 1
                print('Adding to i')

        if (i==50): 
            
            print('WARNING: Randomly assigning code word')
            #Need To assign Random Codeword 
            
            NumGenesInCodebook = len(UsedCodebook)
            NextGene = FPKMS_table['gene_symbols'][NumGenesInCodebook]
            FPKM_toadd = FPKMS_table['avg'][NumGenesInCodebook]
            
            TempDfToFindCodeword = RemainingCodebook.to_numpy()[0]
            ArrayToAppend = list(TempDfToFindCodeword[1:6]) + [NextGene]
            print(ArrayToAppend)
            TempDfToAppend = pd.DataFrame(columns = ['hyb1', 'hyb2', 'hyb3', 'hyb4', 'hyb5', 'Gene'])
            print(TempDfToAppend)
            TempDfToAppend = TempDfToAppend.append({
                'hyb1': ArrayToAppend[0],
                'hyb2': ArrayToAppend[1],
                'hyb3': ArrayToAppend[2],
                'hyb4': ArrayToAppend[3],
                'hyb5': ArrayToAppend[4],
                'Gene': ArrayToAppend[5]
            }, ignore_index = True)
            
            UsedCodebook = pd.concat((UsedCodebook, TempDfToAppend))
            UsedCodebook = UsedCodebook.reset_index(drop = True)
            
            assert(len(UsedCodebook) == NumGenesInCodebook+1)
            
            FPKMS['hyb1'][ArrayToAppend[0]]+= FPKM_toadd
            FPKMS['hyb2'][ArrayToAppend[1]]+= FPKM_toadd
            FPKMS['hyb3'][ArrayToAppend[2]]+= FPKM_toadd
            FPKMS['hyb4'][ArrayToAppend[3]]+= FPKM_toadd
            FPKMS['hyb5'][ArrayToAppend[4]]+= FPKM_toadd
            
            
            RemainingCodebook = RemainingCodebook.drop(RemainingCodebook[(RemainingCodebook['hyb1'] == ArrayToAppend[0])
                & (RemainingCodebook['hyb2'] == ArrayToAppend[1])
                & (RemainingCodebook['hyb3'] == ArrayToAppend[2])
                & (RemainingCodebook['hyb4'] == ArrayToAppend[3])
                & (RemainingCodebook['hyb5'] == ArrayToAppend[4])].index)

            RemainingCodebook = RemainingCodebook.reset_index(drop = True)
         
        counter = counter + 1 
        print(counter)
        
        
    
    UsedCodebook = UsedCodebook.astype({"hyb1":'int', "hyb2":'int', "hyb3":'int', "hyb4":'int', "hyb5":'int'})
    
    print("COMPLETED GENERATING CODEBOOK") 
    return UsedCodebook


def CreateVariableCodewords(pseudocolors):
    '''
    Function to create all possible barcodes for a specific number of pseudocolors.
    
    pseudocolors - Integer number of pseudocolors in codebook design.
    
    Returns an array of barcode arrays. 
    '''
    # Range of possible pseudocolor values 
    Range = []

    for n in np.arange(1.0, pseudocolors + 1.0, 1.0):
        Range.append(n)
    
    Codebook = []
    for q in Range:
        for w in Range: 
            for e in Range: 
                for r in Range: 
                    t = ((q+w+e+r)%len(Range))
                    if (t == 0):
                        t = len(Range)
                    Codebook.append([q,w,e,r,t])
                
    assert(len(Codebook) == pseudocolors**4)
                
    FullCodebook = Codebook.copy()           
    #FullCodebook here is defined as the complete codebook

    return FullCodebook, Codebook           
    
    
def CreateEmptyCodebook(listOfBarcodes):
    '''
    Function to create a pandas dataframe of all possible barcodes. Designed to take output from `CreateVariableCodewords`.

    listOfBarcodes - An array of barcode arrays. E.g. [[1,2,3,4,5], [3,4,5,6,7]]

    Returns a pandas dataframe containing each barcode under an unnamed gene(e.g. 'Gene1') as a placeholder.
    '''
    maindict = dict()
    for i in range(len(listOfBarcodes)):
        gene = f'Gene{i}'
        maindict.update({gene:listOfBarcodes[i]})
    maindict = pd.DataFrame(maindict).T
    maindict.columns = ['hyb1', 'hyb2', 'hyb3', 'hyb4', 'hyb5']
    maindict.index.name = 'Gene'
    maindict = maindict.reset_index(drop = False)

        
    return maindict


def GenerateCodebook(pc, FPKMS):
    '''
    User friendly final function to generate Barcodes for each gene. Generates the barcodes such that they satisfy the parity check ('hyb1' + 'hyb2' + 'hyb3' + 'hyb4')%pseudocolors = 'hyb5'. Any barcode that satisfies this parity check may be assigned to a gene. 
    
    Inputs
    pc - Integer number of pseudocolors in codebook design.
    FPKMS - Pandas dataframe containing the genes under 'gene_symbols', and the expression value under 'avg'.
    
    Returns a pandas dataframe containing all genes and their barcode assignment.
    '''
    FullCodebook, _ = CreateVariableCodewords(pc)
    EmptyCodebook = CreateEmptyCodebook(FullCodebook)
    print('Created empty codebook')
    
    HighExpression = AssignHighExpressionCodewords(FPKMS, 9)
    print('Assigned Highest Expressing Genes')
    
    Channel1Codebook = CreateFullCodebook(HighExpression, EmptyCodebook, FPKMS, pc)
    
    return Channel1Codebook

    
def VerifyCodebook(Codebook):
    '''
    Inputs
    Codebook - pandas dataframe containing all genes, under 'Gene', and their barcode assignment under 'hyb1', 'hyb2', 'hyb3', etc.
    
    Prints statements based on whether all genes are unique and whether all barcodes are unique. 
    '''
    
    CopyFullCodebook = Codebook.copy()
    CopyFullCodebook = CopyFullCodebook.drop_duplicates(subset = ['hyb1', 'hyb2', 'hyb3', 'hyb4', 'hyb5'], keep = 'first')
    
    if (len(Codebook) == len(CopyFullCodebook)):
        print('Codebook has correct length and no repeats')
    else: 
        print('CODEBOOK HAS REPEAT BARCODES')
    
    
    numberOfGenes = Codebook['Gene'].unique()

    if (len(numberOfGenes) == len(Codebook)):
        print('All genes are unique')
    else: 
        print('CODEBOOK HAS REPEAT GENES')
    
    
def numGenesPerPC(FullCodebook, pc):
    '''
    Inputs
    FullCodebook - pandas dataframe containing all genes, under 'Gene', and their barcode assignment under 'hyb1', 'hyb2', 'hyb3', etc.
    pc - Integer number of pseudocolors in codebook design.
    
    Prints number of genes for each Barcoding round('Hyb1', 'Hyb2', 'Hyb3', 'Hyb4', 'Hyb5') and each pseudocolor. 
    '''
    for i in range(1, pc+1):
        print('Hyb1', i, (FullCodebook['hyb1'] == i).sum())

    for i in range(1, pc+1):
        print('Hyb2', i, (FullCodebook['hyb2'] == i).sum())

    for i in range(1, pc+1):
        print('Hyb3', i, (FullCodebook['hyb3'] == i).sum())

    for i in range(1, pc+1):
        print('Hyb4', i, (FullCodebook['hyb4'] == i).sum())

    for i in range(1, pc+1):
        print('Hyb5', i, (FullCodebook['hyb5'] == i).sum())
    