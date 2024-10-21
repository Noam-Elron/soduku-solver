   
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

