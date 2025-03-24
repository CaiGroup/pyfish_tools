from image_stitching import *


#src: directory of your images
src = "/groups/CaiLab/personal/Lex/raw/250203_mb_161genes/pyfish_tools/output/isolated_images2"

#src = "/groups/CaiLab/personal/Lex/raw/250203_mb_161genes/pyfish_tools/output/spatial_mapped_masks/"

#initialize class
IStitch = ImageStitcher(px_size=0.103)

# ## Stitch images with segmentation marker or dapi
IStitch.stitch_images_from_csv(img_dir=src, imgchn=1, stain="dapi", num_channels=2)

## Stitch masks with cell-type definitions
#IStitch.stitch_rgb_images_from_csv(img_dir=src, stain="spatialmapped")

