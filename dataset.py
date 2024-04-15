import os
import numpy as np
import torch
import glob
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *


import os
import numpy as np
import torch
import tifffile
import torch.utils.data

class DatasetLoadAll(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        """
        Initializes the dataset with the path to a folder containing TIFF stacks,
        and an optional transform to be applied to each slice.

        Parameters:
        - root_folder_path: Path to the root folder containing TIFF stack files.
        - transform: Optional transform to be applied to each slice.
        """
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.preloaded_data = {}  # To store preloaded data
        self.slices = self.preload_and_process_stacks()

    def preload_and_process_stacks(self):
        all_slices = []
        for subdir, _, files in os.walk(self.root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
            for filename in sorted_files:
                full_path = os.path.join(subdir, filename)
                stack = tifffile.imread(full_path)
                self.preloaded_data[full_path] = stack  # Preload data here
                for i in range(stack.shape[0]):
                    all_slices.append((full_path, i))
        return all_slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        file_path, slice_index = self.slices[index]
        slice_data = self.preloaded_data[file_path][slice_index, ...]

        # Ensure data has a channel dimension before applying any transform
        if slice_data.ndim == 2:
            slice_data = slice_data[np.newaxis, ...]  # Add channel dimension (C, H, W)

        # Apply the transform if specified
        if self.transform:
            slice_data = self.transform(slice_data)

        return slice_data

# Usage example remains the same as before.


