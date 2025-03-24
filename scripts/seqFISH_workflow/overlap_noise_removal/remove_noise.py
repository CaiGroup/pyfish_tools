"""
@klcolon
date:03.14.25
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

class ChannelNoiseRemover:
    def __init__(self, path: str, radius: float = 0.75, total_hybs: int = 24, total_rounds: int = 4):
        """
        Initializes the noise remover with given parameters.

        Parameters
        ----------
        path : str
            Glob pattern for the files to process.
        radius : float, optional
            Search radius for noise removal (default is 0.75).
        total_hybs : int, optional
            Total number of hybs (default is 24).
        total_rounds : int, optional
            Total number of rounds (default is 4).
        """
        self.path = path
        self.radius = radius
        self.total_hybs = total_hybs
        self.total_rounds = total_rounds
        self.hybs_per_round = int(total_hybs / total_rounds)
        self.rounds = []
        all_hybs = np.arange(0, total_hybs, 1)
        k = 0
        for _ in range(total_rounds):
            self.rounds.append(all_hybs[k:k+self.hybs_per_round])
            k += self.hybs_per_round

    def _find_probable_noise(self, df1: pd.DataFrame, df2: pd.DataFrame) -> np.ndarray:
        """
        Performs a nearest neighbor search to identify probable noise spots.

        If the number of neighboring spots exceeds (hybs_per_round - 2), 
        the spot is considered noise.

        Parameters
        ----------
        df1 : pd.DataFrame
            First set of dots (spots).
        df2 : pd.DataFrame
            Second set of dots (spots).

        Returns
        -------
        np.ndarray or None
            An array of indices in df1 that are considered noise, or None if no noise is found.
        """
        # Reset index for safety
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)

        if df2.empty:
            print("Warning: DataFrame is empty!")
            return None
        
        # Initialize and fit the nearest neighbors model
        neigh = NearestNeighbors(n_neighbors=2, radius=self.radius, metric="euclidean", n_jobs=1)
        initial_seed = df1[["x", "y"]]
        neigh.fit(df2[["x", "y"]])
        _, neighbors = neigh.radius_neighbors(initial_seed, self.radius, return_distance=True, sort_results=True)
        
        # Identify noise spots based on neighbor count threshold
        neighbors_flattened = []
        for i in range(len(neighbors)):
            try:
                if len(neighbors[i]) > (self.hybs_per_round - 2):
                    neighbors_flattened.append([i, neighbors[i]])
            except IndexError:
                continue
        
        if len(neighbors_flattened) == 0:
            return None
        else:
            return np.array(neighbors_flattened, dtype=object)[:, 0]

    def process_file(self, file: str):
        """
        Processes a single file, removing noise spots within each barcode round,
        and writes the cleaned data to a new CSV file.

        Parameters
        ----------
        file : str
            Path to the CSV file to process.
        """
        filename = Path(file).name
        output_path = str(Path(file).parent / f"noise_removed_{filename}")
        df = pd.read_csv(file)
        new_df_list = []
        
        for barcode_round in self.rounds:
            # Subset the dataframe for the current round
            df_hyb = df[df.hyb.isin(barcode_round)].reset_index(drop=True)
            # Identify indices to remove as noise
            remove = self._find_probable_noise(df_hyb, df_hyb)
            if remove is not None:
                cleaned_df = df_hyb.drop(remove)
                new_df_list.append(cleaned_df)
            else:
                new_df_list.append(df_hyb)
        
        # Concatenate cleaned rounds and write to CSV
        new_df = pd.concat(new_df_list).reset_index(drop=True)
        new_df.to_csv(output_path, index=False)
        print(f"Processed file: {file}")

    def process_all_files(self):
        """
        Processes all files matching the given path pattern.
        """
        all_files = glob.glob(self.path)
        for file in tqdm(all_files):
            self.process_file(file)


# Example usage:
if __name__ == "__main__":
    path = "/groups/CaiLab/personal/Lex/raw/250310_150_nih3t3_current/pyfish_tools/output/dots_detected/Channel_1/spots_in_cells/spots_norm/*/locations_z_*"
    remover = ChannelNoiseRemover(path, radius=0.75, total_hybs=24, total_rounds=4)
    remover.process_all_files()