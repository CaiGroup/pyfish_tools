# Welcome to pyFISH!
<p align="center">
<img src="https://github.com/CaiGroup/pyfish_tools/blob/pyFISH/logo/logo.svg" alt="fish icon" width="400" height="200">
</p>
PyFISH is a generalized, fully Python-based pipeline for the analysis of various single-molecule FISH (smFISH) methods. The pipeline also includes a decoder for various sequential FISH (seqFISH) encoding schemes. It is recommended that users utilize a high-performance cluster (HPC), since the pipeline is already formatted to use the slurm workload manager. Sbatch file templates can be found in each directory to meet the user's needs. Current pipeline wrapped around a NextFlow workflow manager will be availible soon!

## General Workflow
<p align="center">
<img src="https://github.com/CaiGroup/pyfish_tools/blob/pyFISH/logo/workflow.png" alt="pipeline">
</p>

## Setting up your Conda environment

To set up your Conda environment, please run the following yml file: `conda env create -f python3.7.yml`

## Copying the pipeline folder

To begin using pyFISH tools, copy the pipeline folder into your raw images directory by running the following command: `cp -r /path/to/pyfish_tools /path/to/raw/images/`

Outputs will be generated into a directory called "output" within the pyfish_tools directory.

## What is included in this pipeline?
### Alignment
- Align images with phase cross correlation using DAPI
- Align images with RANSAC adjusted affine transformation using fiducial beads 
- Align z shifts with normalized correlation analysis
### Chromatic Aberration Correction
- Align channels with RANSAC adjusted affine transformation by setting one channel as reference
### Image Processing
- Richardson-Lucy deconvolution using various point spread functions
- High pass Gaussian filter to isolate signal (with added match histrogram function for very noisy images)
- Low pass Gaussian filter to smooth out single molecule spots or remove hot pixels 
- Background subtraction using empty images
- Gamma enhancement to boost signal
- Rolling ball subtraction to even illumination while simultaneously isolating true signal
- Division by 1D Gaussian convolution for evening out image illumination
- TopHat to remove large blobs such as lipofuscin 
- Scaling intensities of images by percentile clipping
### Spot Detection
- Detection of single molecule spots across all z-planes and channel axes using DAOStarFinder  
- Ratiometric dot detection to determine the contribution of various fluorophores in a given spot
### Segmentation
- Generate cell masks using Cellpose 2.0
- Keep cytoplasm masks that also overlap with nuclear masks
- Stitch nuclear and cytoplasm masks so that two separate sub-cellular analysis can be performed
### Cell Mask Borders and Edge Deletion
- Delete n number of pixels for two or more masks that touch
- Delete masks that are at the image borders
### Spot Mapping
- For detected spots, you can map them to their corresponding cell masks
### Decoding
- Gene assignment of non-barcoded sequential smFISH spots
- Gene assignment of barcoded sequential smFISH spots with SVM embedded, feature-based nearest neighbor radial decoder
### Gene by Cell 
- Generate gene-by-cell matrix for single-cell analysis
### Noise and Fiducial Removal
- Remove fiducial markers or background spot-like noise by spot detecting on background image and searching for corresponding spot in subsequent images
- Remove fiducial markers or background spot-like noise by looking for redudant spots within a barcoding round
### Codebook Generation
- Convert pseudocolor codebooks to an n-bit string codebook
- Balance codebook using TPM or FPKM values from RNA-seq
### Image Stitching
- Stitch images based on metadata to generate one composite image
- Find position of cells across FOVs
### Density Estimations
- Calculate optical density of spots
### Colocalization Assessment
- Calculate the colocalization efficiency between two or more spots
### Post Analysis
- Calculate final false positive rate of decoded barcodes
- Perform correlational analysis with other smFISH datasets or RNA-seq
- Basic clustering 







