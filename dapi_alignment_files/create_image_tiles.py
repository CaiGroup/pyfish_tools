from util import pil_imread

def split_image(img_src, tile_portion=0.1):
    """
    Function to split large images into tiles.
    
    Parameters
    ----------
    img_src: image path
    tile_portion: the size proportion of tiles
    """
    assert tile_portion < 1, print("Tile portion must be less than 1.")
    
    #tiles
    tiles = int(1/tile_portion)
    
    #read image
    img = pil_imread(img_src, swapaxes=True)
    
    #get shape of image
    xy_shape = img[0,0,:,:].shape
    
    #get x and y intervals to get desired tiles
    x_slice = xy_shape[0]//tiles
    y_slice = xy_shape[1]//tiles
    
    #generate x and y cut array
    x_cut = [x for x in range(0,xy_shape[0],x_slice)]
    y_cut = [y for y in range(0,xy_shape[1],y_slice)]
    
    #make tiles
    img_slice = []
    for x in x_cut:
        for y in y_cut:
            print(f"slice = {x}:{x+x_slice}, {y}:{y+y_slice}")
            img_slice.append(img[:,:,x:x+x_slice,y:y+y_slice])
    
    return img_slice
