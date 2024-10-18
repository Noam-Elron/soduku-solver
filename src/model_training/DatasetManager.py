import os
import glob 
from typing import List, Tuple, Union

class DatasetManager:
    def __init__(self, directory):
        self.cur_directory = directory
        self.cwd = None
        self.path = None
        self.set_paths()

    def set_paths(self):
        self.cwd = os.getcwd()
        path = os.path.join(self.cwd, self.cur_directory)
        # Normalize filepath to work for both windows and linux
        self.path = os.path.normcase(path) 

    def get_all_img_files(self):
        """Returns all images from the dataset, image file_extension must be jpg,jpeg or png.

        Args:
            None

        Returns:
            filepaths: List of paths to each image in the directory.
        """
        images = glob.glob(os.path.join(self.path, "*.jpg")) + glob.glob(os.path.join(self.path, "*.jpeg")) + glob.glob(os.path.join(self.path, "*.png"))
        return images