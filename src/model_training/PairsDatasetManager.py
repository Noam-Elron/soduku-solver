import os
import glob 
from typing import List, Tuple, Union
import re
from .DatasetManager import DatasetManager

class ImgDatDirectoryManager(DatasetManager):
    def __init__(self, directory):
        super().__init__(directory)

    def get_all_img_files(self, reverse: bool = False):
        """Returns all images from the dataset, expects images to be in the format image#number.file_extension where file_extension is either jpg,jpeg or png.

        Args:
            reverse (bool, optional): Flag to determine if images are returned in ascending #number order or descending. Defaults to False which is Ascending.

        Returns:
            filepaths: List of paths to each image in the directory.
        """
        images = glob.glob(os.path.join(self.path, "*.jpg")) + glob.glob(os.path.join(self.path, "*.jpeg")) + glob.glob(os.path.join(self.path, "*.png"))
        # Sorting the files based on number.
        reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\"  else r"(?<=\/image)[0-9]+(?=.)"
        try:
            images = sorted(images, key = lambda img: int(re.search(reg_exp, img).group()), reverse=reverse)
        except AttributeError:
            raise AttributeError("Incompatible/misnamed file found in directory")
        
        return images
    
    def get_all_dat_files(self, reverse: bool = False):
        """Returns all .dat from the dataset, expects dat to be in the format image#number.dat

        Args:
            reverse (bool, optional): Flag to determine if images are returned in ascending #number order or descending. Defaults to False which is Ascending.

        Returns:
            filepaths: List of paths to each .dat file in the directory.
        """
        data = glob.glob(os.path.join(self.path, "*.dat"))
        # Sorting the files based on number.
        reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\"  else r"(?<=\/image)[0-9]+(?=.)"
        try:
            data = sorted(data, key = lambda dat: int(re.search(reg_exp, dat).group()), reverse=reverse)
        except AttributeError:
            raise AttributeError("Incompatible/misnamed file found in directory")
        
        return data

    def get_all_img_dat_files(self, reverse: bool = False) -> List[Tuple[str, str]]:
        """
        Retrieves all image, data file pairs. \n
        Returns:
            2D Array where each element is a Tuple that contains filepaths for a IMAGE,DATA pair.
        Raises:
            AttributeError - Incompatible/Misnamed file in given directory 
        """
        imgs = self.get_all_img_files(reverse)
        data = self.get_all_dat_files(reverse)

        pairs = list(zip(imgs, data))
        return pairs

    def get_image_labels(self, data: str) -> List[str]:
        """
        Return all the sudoku_numbers belonging to an associated sudoku board image.
        
        Returned is a 1D array with the numbers being ordered by row left to right.
        So every nine numbers a row is passed, rows begin from top to bottom.

        Parameters:
            data(str) -- filepath to a .dat file 
        Returns:
            1D Array of digits, each index of the array is mapped to a position in a sudoku board.
        """
        values = []
        with open(data, "r") as data:
            rows = data.readlines()
            for row in rows:
                # Remove the \n using the strip method and since all the digits are whitespace separated strings, split them using the split method using a " " - whitespace separator
                digits = row.strip().split(" ")
                values.append(digits)
        # values array: 2D array, where each list inside it is a row from the sudoku board, and each such row contains all digits belonging to the appropriate cell in that row. 
        # Nested list comprehension to unpack the values array into a 1D array of just digits. Easier to manipulate afterwards
        output = [digit for row in values for digit in row]
        return output
    
    def normalize_file_names(self, bottom_range=0):
        """
        Renames all existing files to img#, data# pairs 
        Parameters:
            bottom_range(int) - Starting count for what number onwards to name the files.
        Returns:
            2D Array where each element is a Tuple that contains filepaths for a IMAGE,DATA pair.
        Raises:
            AttributeError - Incompatible/Misnamed file in given directory 
        """
        files = glob.glob(os.path.join(self.path, "*"))      
        img_number = bottom_range
        dat_number = bottom_range
        for file in files:
            _, file_extension = os.path.splitext(file)
            #print(file, file_extension)
            
            if file_extension in [".jpg", ".jpeg", ".png"]:
                os.rename(file, f"{self.path}\image{img_number}.{file_extension}")
                #print(f"{self.path}\image{img_number}{file_extension}")
                img_number += 1
            elif file_extension == ".dat":
                os.rename(file, f"{self.path}\image{dat_number}.dat")
                #print(f"{self.path}\image{img_number}{file_extension}")
                dat_number += 1
            else:
                raise AttributeError("Incompatible/misnamed file found in directory")
    

    


 