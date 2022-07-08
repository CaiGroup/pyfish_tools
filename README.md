# seqFISH Datapipeline
Generalized and fully python-based pipeline for the analysis of various seqFISH methodologies.

## Important Dependencies
- python >= 3.7 (3.7 preferred)

## Setup conda enviroment by running the following command:
```
conda env create -f python3.7.yml
```

## webfish_tools soft link
Go to your python enviroment 
```
cd ~/miniconda3/envs/python3.7/lib/python3.7/site-packages
```
create a soft link with the following
```
ln -s /groups/CaiLab/personal/python_env/lib/python3.9/site-packages/webfish_tools ./
```

### For now, just copy the scripts into a folder named notebook_pyfiles in your raw images directory. 
```
cp -r /path/to/seqFISH_datapipeline /path/to/raw/images/notebook_pyfiles
```
### Main Developers and Contributions (Go to them for any questions)
- Katsuya Lex Colon
	- Overall pipeline structure (batch files and job submission)
	- Dapi alignment scripts
	- Chromatic aberration correction scripts (collaboration with Lincoln Ombelets)
	- Fiducial alignment scripts
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
