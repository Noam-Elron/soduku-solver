   
from typing import List, Tuple, Union     
import os 
import glob

def reformat_data(dat: str) -> List[str]:
        """
        Reformat .dat file with additional lines to proper format.
        
        Parameters:
            dat(str) -- filepath to a .dat file 

        Returns:
            Nothing
        """
        
        values = []

        with open(dat, "r") as data:
            lines = data.readlines()
            if (len(lines) == 9):
                 return
            # Ignore first two lines
            for line in lines[2:]:
                # Remove the \n using the strip method and since all the digits are whitespace separated strings, split them using the split method using a " " - whitespace separator
                digits = line.strip() + "\n"
                values.append(digits)
        # values array is a 2d array where each of its elements is an array - "row" of chars. Double list comprehension to unpack the values array into a 1D array of just digits. Easier to manipulate afterwards. 
        with open(dat, "w") as data:
             data.writelines(values)

def normalize_file_names(directory: str, bottom_range=0):
        """
        Renames all existing files to img#, data# pairs 
        Parameters:
            bottom_range(int) - Starting count for what number onwards to name the files.
        Returns:
            2D Array where each element is a Tuple that contains filepaths for a IMAGE,DATA pair.
        Raises:
            AttributeError - Incompatible/Misnamed file in given directory 
        """
        cwd = os.getcwd()
        path = os.path.join(cwd, directory)
        # Normalize filepath to work for both windows and linux
        path = os.path.normcase(path)
        files = glob.glob(os.path.join(path, "*"))      
        img_number = bottom_range
        dat_number = bottom_range
        for file in files:
            _, file_extension = os.path.splitext(file)
            #print(file, file_extension)
            
            if file_extension in [".jpg", ".jpeg", ".png"]:
                os.rename(file, f"{path}\image{img_number}.{file_extension}")
                print(f"{path}\image{img_number}{file_extension}")
                img_number += 1
            elif file_extension == ".dat":
                os.rename(file, f"{path}\image{dat_number}.dat")
                print(f"{path}\image{img_number}{file_extension}")
                dat_number += 1
            else:
                raise AttributeError("Incompatible/misnamed file found in directory")