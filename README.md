# seqFISH Datapipeline
Generalized and fully python-based pipeline for the analysis of various seqFISH methodologies.

## Important Dependencies
- python >= 3.7 (3.7 preferred)

### For now, just copy the scripts into a folder named notebook_pyfiles in your raw images directory. 
```
cp -r /path/to/seqFISH_datapipeline /path/to/raw/images/notebook_pyfiles
```
### Main Developers and Contributions
- Katsuya Lex Colon
	-Overall pipeline structure (batch files and job submission)
	-Dapi alignment scripts
	-Chromatic aberration correction scripts (collaboration with Lincoln Ombelets)
	-Fiducial alignment scripts
	-Fiducial removal scripts
	-Dot detection scripts
	-Density estimation scripts
	-Colocalization scripts
	-Edge deletion scripts
	-Gene mapping scripts
	-Gene by cell scripts
	-Decoding scripts
	-Post analysis scripts
	-Codebook converter script
	-Mask generation scripts (collaboration with Arun Chakravorty)
	-Pre-processing scripts (collaboration with Shaan Sekhon, Michal Polonsky, and Anthony Linares)
- Arun Chakravorty
	-Codebook balancer scripts
	-Mask generation scripts (collaboration with Katsuya Lex Colon)
	-image stitching scripts (collaboration with Lincoln Ombelets)
- Lincoln Ombelets
	-Chromatic aberration correction scripts (collaboration with Katsuya Lex Colon)
	-Image stitching scripts (collaboration with Arun Chakravorty)
	-Util.py script for consistent image reading
