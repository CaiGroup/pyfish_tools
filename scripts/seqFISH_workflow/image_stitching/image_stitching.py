"""
@klcolon
Date: 250117
"""

import json
import pandas as pd
import numpy as np
import tifffile as tf
import cv2
from tqdm import tqdm
from pathlib import Path
import os
import warnings
from util import pil_imread
from typing import Tuple

# Ignore all warnings
warnings.filterwarnings('ignore')

class ImageStitcher:
    """
    A class to handle position file conversion and image stitching for microscopy data.
    """

    def __init__(self, px_size: float = 0.11):
        """
        Initializes the ImageStitcher with a given pixel size.

        :param px_size: Pixel size in micrometers.
        """
        self.px_size = px_size

    def convert_pos_file_to_csv(self, img_path: str, output_file_name: str = None) -> Tuple[pd.DataFrame, Path]:
        """
        Parses stage positions from a JSON `.pos` file and converts them to a CSV format.

        :param img_path: Path to the directory containing the `.pos` file.
        :param output_file_name: Optional output file name for the stitched image.
        :return: A DataFrame with position information and the output file path.
        """
        parent = Path(img_path)
        while "pyfish_tools" not in os.listdir(parent):
            parent = parent.parent

        output_file = parent / "pyfish_tools" / "output" / "stitched_img" / (output_file_name + "_stitched.tif")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        filename = str(list(parent.glob("*.pos"))[0])
        with open(filename) as f:
            content = json.load(f)

        positions = []
        for pos in content["map"]['StagePositions']["array"]:
            label = pos["Label"]["scalar"]
            cor = pos['DevicePositions']["array"]

            index = 1 if cor[0]["Device"]["scalar"] == 'Adaptive Focus Control Offset' else 0
            coordinates = cor[index]['Position_um']["array"]

            if len(coordinates) < 2:
                print(f"Warning: Position {label} has incomplete coordinates. Skipping.")
                continue

            posinfo = {
                'label': label,
                'x': coordinates[0],
                'xpx': round(coordinates[0] / self.px_size),
                'y': coordinates[1],
                'ypx': round(coordinates[1] / self.px_size)
            }
            positions.append(posinfo)

        return pd.DataFrame(positions), output_file

    def stitch_images_from_csv(self, img_dir: str, imgchn: int = 0, stain: str = "dapi", num_channels: int = 2) -> None:
        """
        Stitches microscopy image tiles into a single image using position data.

        :param img_dir: Directory containing image tiles.
        :param imgchn: Image channel to use for stitching.
        :param stain: Stain type used for naming the output file.
        :param num_channels: Number of channels in the image tiles.
        """
        pos_df, output_file_name = self.convert_pos_file_to_csv(img_dir, stain)
        pos_df["label"] = pos_df.index

        tile_size = 2048
        min_x, min_y = pos_df['xpx'].min(), pos_df['ypx'].min()
        max_x, max_y = pos_df['xpx'].max(), pos_df['ypx'].max()
        width, height = (max_x - min_x) + tile_size, (max_y - min_y) + tile_size

        stitched_image = np.zeros((height, width), dtype=np.uint16)

        for _, row in tqdm(pos_df.iterrows(), total=pos_df.shape[0]):
            label, xpx, ypx = int(row['label']), int(row['xpx'] - min_x), int(row['ypx'] - min_y)
            image_path = Path(img_dir) / f"MMStack_Pos{label}.ome.tif"

            if image_path.exists():
                img_tile = pil_imread(str(image_path), num_channels=num_channels)
                img_tile = img_tile[imgchn] if img_tile.ndim == 3 else np.max(img_tile, axis=1)[imgchn]
                stitched_image[ypx:ypx + tile_size, xpx:xpx + tile_size] = img_tile
            else:
                print(f"Warning: {image_path} not found. Skipping.")

        tf.imwrite(output_file_name, stitched_image, compression='DEFLATE')

    def stitch_rgb_images_from_csv(self, img_dir: str, stain: str = "dapi") -> None:
        """
        Stitches RGB microscopy image tiles into a single image using position data.
    
        :param img_dir: Directory containing image tiles.
        :param stain: Stain type used for naming the output file.
        """
        pos_df, output_file_name = self.convert_pos_file_to_csv(img_dir, stain)
        pos_df["label"] = pos_df.index
    
        tile_size = 2048
        min_x, min_y = pos_df['xpx'].min(), pos_df['ypx'].min()
        max_x, max_y = pos_df['xpx'].max(), pos_df['ypx'].max()
        width, height = (max_x - min_x) + tile_size, (max_y - min_y) + tile_size
    
        # Initialize the stitched image with three channels for RGB
        stitched_image = np.zeros((height, width, 3), dtype=np.float64)
    
        for _, row in tqdm(pos_df.iterrows(), total=pos_df.shape[0]):
            label, xpx, ypx = int(row['label']), int(row['xpx'] - min_x), int(row['ypx'] - min_y)
            image_path = Path(img_dir) / f"MMStack_Pos{label}.ome.tif"
    
            if image_path.exists():
                img_tile = tf.imread(str(image_path))
                if img_tile.ndim == 3 and img_tile.shape[2] == 3:  # Ensure the tile is RGB
                    stitched_image[ypx:ypx + tile_size, xpx:xpx + tile_size, :] = img_tile
                else:
                    print(f"Warning: {image_path} is not an RGB image. Skipping.")
            else:
                print(f"Warning: {image_path} not found. Skipping.")
                
        stitched_image = (stitched_image / stitched_image.max() * 255).astype('uint8')  # Normalize and convert to uint8
        tf.imwrite(output_file_name, stitched_image, photometric='rgb', compression='DEFLATE')