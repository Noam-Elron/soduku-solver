from fileprompter import FileDialogWindow
import cv2 as cv
import numpy as np
import os
from utils import multi_image_show_matplotlib, pad_image, image_show_matplotlib, return_cells
import glob 
import re
from typing import List, Tuple, Union
import pandas as pd


"""

This file is simply for readying data taken from an online repository which contains pairs of sudoku boards images 
along with a data file that contains a string version of whats inside the file.

"""


def get_all_data_pairs(directory: str = "dataset") -> List[Tuple[str, str]]:
    """
    Retrieves all image, data file pairs.

    Parameters:
        directory(str) - default = "dataset" -- name of folder containing image,data pairs

    Returns:
        2D Array where each element is a Tuple that contains filepaths for a IMAGE,DATA pair.

    Raises:
        AttributeError - Incompatible/Misnamed file in given directory 
    """
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, directory)
    # Normalize filepath to work for both windows and linux
    path = os.path.normcase(path) 
    imgs = []
    data = []
    # Create a list of all files in given directory path
    files = glob.glob(os.path.join(path, "*"))
        
    for file in files:
        _, file_extension = os.path.splitext(file)
        if file_extension not in [".jpg", ".jpeg", ".png", ".dat"]:
            raise AttributeError("Incompatible/misnamed file found in directory")
        if file_extension in [".jpg", ".jpeg", ".png"]:
            imgs.append(file)
        elif file.lower().endswith(".dat"):
            data.append(file)
        
    pairs = list(zip(imgs, data))
    return pairs


def label_cells(cells: List[List[int]], labels: List[str]) -> List[Union[str, int]]:
    """
    Adds a label to each cell image. 

    Parameters:
        cells(List[List[int]]) -- 2D Array containing 81 images of cells, each image of size 28*28. \n
        labels(List[str]) -- Array of digits in string form, each digit corresponds corresponds to the digit appearing in the corresponding cell position in the cells list. 

    Returns:
        2D Array where each element is an array, the first element in the array is a string that represents the digit appearing and the rest are the flattened pixels of the cell image. 
    """
    return [[labels[i]] + cell.tolist() for i, cell in enumerate(cells)]

def read_dat(data: str) -> List[str]:
    """
    Return all the digits belonging to the associated sudoku board image. Used to normalize/convert the original .dat files into an easier format

    Parameters:
        data(str) -- filepath to a .dat file 

    Returns:
        1D Array of digits, each index of the array is mapped to a position in a sudoku board.
    """
    
    values = []
    with open(data, "r") as data:
        lines = data.readlines()
        # Ignore first two lines
        for line in lines:
            # Remove the \n using the strip method and since all the digits are whitespace separated strings, split them using the split method using a " " - whitespace separator
            digits = line.strip().split(" ")
            values.append(digits)
    # values array is a 2d array where each of its elements is an array - "row" of chars. Double list comprehension to unpack the values array into a 1D array of just digits. Easier to manipulate afterwards. 
    output = [digit for row in values for digit in row]
    return output


def return_all_data() -> List[Union[str, int]]:
    """
    Combines every single cell with a label

    Parameters:
        None

    Returns:
        Returns a 2D array where each element is a list containing a label and 784 which make up an image of a 28*28 cell
    
    Raises:
        Nothing
    """

    df_data = []
    pairs = get_all_data_pairs("dataset")
    for img, data in pairs:
        labels = read_dat(data)
        cells = return_cells(img)
        df_data += label_cells(cells, labels)

    return df_data
    # List comprehension way just to test, plus in case in need of speed up as list comprehension is almost always much faster than append/concat
    #return [labled_cell for img, data in pairs for labled_cell in label_cells(return_cells(img), read_dat(data))]

def create_df(all_data: List[Union[str, int]]):
    """
    Creates pandas dataframe using given data

    Parameters:
        all_data - 2D array of elements that contain a label and 784 pixels.

    Returns:
        Pandas dataframe object
    
    Raises:
        Nothing
    """

    row_names = ["label"] + [f"pixel{i}" for i in range(0, 28*28)]
    df = pd.DataFrame(all_data, columns=[*row_names])
    return df

def reformat_data(data: str) -> List[str]:
    """
    Return all the digits belonging to the associated sudoku board image. 

    Parameters:
        data(str) -- filepath to a .dat file 

    Returns:
        1D Array of digits, each index of the array is mapped to a position in a sudoku board.
    """
    
    values = []
    with open(data, "r") as data:
        lines = data.readlines()
        # Ignore first two lines
        for line in lines[2:]:
            # Remove the \n using the strip method and since all the digits are whitespace separated strings, split them using the split method using a " " - whitespace separator
            digits = line.strip().split(" ")
            values.append(digits)
    # values array is a 2d array where each of its elements is an array - "row" of chars. Double list comprehension to unpack the values array into a 1D array of just digits. Easier to manipulate afterwards. 
    output = [digit for row in values for digit in row]
    return output

def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Return combined dataframe

    Parameters:
        df1(pd.Dataframe): Dataframe to be at the beginning of the new Dataframe
        df2(pd.Dataframe): Dataframe to be at the end of the new Dataframe
    Returns:
        New combined Dataframe
    """

    return pd.concat([df1, df2], axis=0)

def rename_files(directory, bottom_range):
    """
    Retrieves all image, data file pairs.

    Parameters:
        directory(str) - default = "dataset" -- name of folder containing image,data pairs

    Returns:
        2D Array where each element is a Tuple that contains filepaths for a IMAGE,DATA pair.

    Raises:
        AttributeError - Incompatible/Misnamed file in given directory 
    """
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, directory)
    # Normalize filepath to work for both windows and linux
    path = os.path.normcase(path) 
    
    files = glob.glob(os.path.join(path, "*"))
    """
    reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\" else r"(?<=\/image)[0-9]+(?=.)"
    #print(reg_exp, "\n", os.sep)
    try:
        files = sorted(files, key = lambda file: int(re.search(r"(?<=(\\|\/)image)[0-9]+(?=.)", file).group()))
    except AttributeError:
        raise AttributeError("Incompatible/misnamed file found in directory")
    """
    
    img_number = bottom_range
    dat_number = bottom_range
    for file in files:
        _, file_extension = os.path.splitext(file)
        print(file, file_extension)
        if file_extension in [".jpg", ".jpeg", ".png"]:
            os.rename(file, f"dataset\image{img_number}.jpg")
            img_number += 1
        elif file_extension == ".dat":
            os.rename(file, f"dataset\image{dat_number}.dat")
            dat_number += 1
    


def output_csv(directory, name, combined=False):
    """
    Output new csv dataset.

    Parameters:
        None

    Returns:
        None, outputs csv file.
    """

    data = return_all_data()
    df = create_df(data)
    df = combine_dataframes(pd.read_csv('./input/train.csv'), df) if combined else df
    df.to_csv(f'./{directory}/{name}.csv', index=False)


def fix_data_files() -> None:
    """
    Helper function to remove first two lines from .dat files as the downloaded .dat files came with two lines of useless info. 

    Parameters:
        None

    Returns:
        None, rewrites files.
    """

    pairs = get_all_data_pairs("dataset")
    for img, data in pairs:
        with open(data, "r+") as file:
            lines = file.readlines()
            file.seek(0)
            file.truncate()
            file.writelines(lines[2:])





def extend_dataset():
    pass
    # TODO, iterate over every file in dataset directory, rotate the image 90 degrees 3 times and each time append that
    # To a new folder - ext_dataset. Also rename each file to be image##-1, image##-2 etc etc.



def main():
    win = FileDialogWindow()
    #make_cells(win.filename)
    #output_csv("input", "custom_combined", combined=True)
    


if __name__ == "__main__":
    main()