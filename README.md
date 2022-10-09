# seqFISH Datapipeline
Generalized and fully python-based pipeline for the analysis of various seqFISH methodologies. Talk to Lex on how to submit jobs and edit batch files.

## Important Dependencies
- python >= 3.7 (3.7 preferred)

## Setup conda enviroment
```
conda env create -f python3.7.yml
```

### Copy the pipeline folder into your raw images directory. 
```
cp -r /path/to/seqFISH_datapipeline /path/to/raw/images/
```
Outputs would be generated into a directory called "output" in seqFISH_datapipeline directory. 

If you want something added to this github repo, let Katsuya Lex Colon know.

### Main Developers and Contributions (Go to them for any questions)
- Katsuya Lex Colon
	- Overall pipeline structure (batch files and job submission)
	- Dapi alignment scripts
	- Chromatic aberration correction scripts (collaboration with Lincoln Ombelets)
	- Fiducial alignment scripts
	- Z alignment scripts
	- Fiducial removal scripts
	- Dot detection scripts
	- Density estimation scripts
	- Colocalization scripts
	- Edge deletion scripts
	- Gene mapping scripts
	- Gene by cell scripts
	- Decoding scripts
	- Post analysis scripts
	- Codebook converter script
	- Requested scripts
	- Mask generation scripts (collaboration with Arun Chakravorty)
	- Pre-processing scripts (collaboration with Shaan Sekhon, Michal Polonsky, and Anthony Linares)
- Arun Chakravorty
	- Codebook balancer scripts
	- Mask generation scripts (collaboration with Katsuya Lex Colon)
	- Image stitching scripts (collaboration with Lincoln Ombelets)
- Lincoln Ombelets
	- Chromatic aberration correction scripts (collaboration with Katsuya Lex Colon)
	- Image stitching scripts (collaboration with Arun Chakravorty)
	- Util.py script for consistent image reading
