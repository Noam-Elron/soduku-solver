import os
import glob 
from typing import List, Tuple, Union
import re
from DatasetManager import DatasetManager

class CellDatasetManager(DatasetManager):
    def __init__(self, directory):
        super().__init__(directory) 

    def get_specific_number_images(self, number: int):
        """Returns all images belonging to a specific digit

        Args:
            number (int): Digit we want the returned images to display.
        Returns:
            List(str): List of filepaths to each image of the specified number.
        """
        number_path = os.path.join(self.path, str(number))
        images = glob.glob(os.path.join(number_path, "*.jpg")) + glob.glob(os.path.join(number_path, "*.jpeg")) + glob.glob(os.path.join(number_path, "*.png"))
        return images
    
    def get_all_img_files(self):
        """Returns all images from the dataset, expects images to be in the format image#number.file_extension where file_extension is either jpg,jpeg or png.

        Args:
            reverse (bool, optional): Flag to determine if images are returned in ascending #number order or descending. Defaults to False which is Ascending.

        Returns:
            List(str): List of filepaths to each image in the directory.
        """
        
        # Sorting the files based on number.
        
        images = []
        for i in range(10):
            images += self.get_specific_number_images(i)
        
        return images