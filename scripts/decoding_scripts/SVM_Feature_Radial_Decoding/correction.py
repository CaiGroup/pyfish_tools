#check each codeword on whether it passes parity 
for i,codes in enumerate(sorted_codes):
    potential = (locations.iloc[list(codes)].sort_values("hyb")["hyb"].values) + 1
    pseudocolors = int(hybs/num_barcodes)
    #convert into pseudocolor sequence
    get_barcode = []
    for k,ps in enumerate(potential):
        get_barcode.append(ps - (k*pseudocolors))\
    #if a barcode is missing a pseudocolor then we can't do parity check
    if len(get_barcode)==(num_barcodes-1):
        continue
    parity_code = (sum(get_barcode[:len(get_barcode)-1]) % pseudocolors)
    if parity_code == 0:
        parity_code=pseudocolors
    #if the codeword passes parity then end loop
    elif  == get_barcode[-1]:
        best = codes
        best_score = sorted_scores[i]
        best_dist = sorted_dist[i]
        break
