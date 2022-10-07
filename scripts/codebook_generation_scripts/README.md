# Generating codebooks
If you have RNA-seq data, use the codebook equalizer to balance hybs based on FPKM or TPM. If you do not have RNA-seq files, then make a normal codebook and assign barcodes to genes randomly. Make sure that the parity code is generated as such, sum(Hyb1...HybN) mod total pseudocolors, where N is total rounds - 1. 

Once the codebooks are created, run the codebook converter script which will be used for decoding. 
