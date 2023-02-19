from fileprompter import FileDialogWindow
from sudoku_scanner import SudokuImage
import cv2 as cv
import numpy as np
import os
from utils import multi_image_show_matplotlib, pad_image, image_show_matplotlib
import glob 
import re
from typing import List, Tuple, Union
import pandas as pd

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
    files = glob.glob(os.path.join(path, "*"))
    #print(files)
    reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\" else r"(?<=\/image)[0-9]+(?=.)"
    #print(reg_exp, "\n", os.sep)
    try:
        files = sorted(files, key = lambda file: int(re.search(r"(?<=(\\|\/)image)[0-9]+(?=.)", file).group()))
    except AttributeError:
        raise AttributeError("Incompatible/misnamed file found in directory")
    for file in files:
        if file.lower().endswith(".jpg"):
            imgs.append(file)
        elif file.lower().endswith(".dat"):
            data.append(file)
    
    pairs = list(zip(imgs, data))
    return pairs

def return_cells(filename: str) -> List[List[int]]:
    """
    Returns all cells extracted from an image
    
    Parameters:
        filename(str): path to an image(specifically a sudoku image)

    Returns:
        2D array of shape [81, 784] 
    """
    img = SudokuImage(filename)
    cells = img.return_all_cells()
    # Resize all cells to be a 28x28 image to be uniform with 
    #multi_image_show_matplotlib(cells, 20, 4)
    cells = [cv.resize(cells[i], (18,18)) for i in range(len(cells))]  
    cells = [pad_image(cell, 5, 0) for cell in cells]
    #multi_image_show_matplotlib(cells, 20, 4)
    cells = np.reshape(cells, (-1, 28*28)) 
    return cells

def label_cells(cells: List[List[int]], labels: List[str]) -> List[Union[str, int]]:
    """
    Adds a label to each cell image. 

    Parameters:
        cells(List[List[int]]) -- 2D Array containing 81 1D array of ints that make up the pixels of a 28*28 image of a given cell. \n
        labels(List[str]) -- Array containing a string representation of the digit, corresponding to a cell that appears in the list of cells. 

    Returns:
        2D Array where each element is an array that contains the label and all the cell pixels. 
    """
    return [[labels[i]] + cell.tolist() for i, cell in enumerate(cells)]

def read_dat(data: str) -> List[str]:
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
        for line in lines:
            # Remove the \n using the strip method and since all the digits are whitespace separated strings, split them using the split method using a " " - whitespace separator
            digits = line.strip().split(" ")
            values.append(digits)
    # values array is a 2d array where each of its elements is an array - "row" of chars. Double list comprehension to unpack the values array into a 1D array of just digits. Easier to manipulate afterwards. 
    output = [digit for row in values for digit in row]
    return output

def look_at_dataset(directory: str):
    """
    Shows every image's warped perspective that is inside the given directory

    Parameters:
        directory(str): directory that contains images whose names start in image followed by a number

    Returns:
        Nothing
    
    Raises:
        Nothing
    """
    import glob
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, directory)
    # Normalize filepath to work for both windows and linux
    path = os.path.normcase(path) 
    files = glob.glob(os.path.join(path, "*.jpg"))
    reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\"  else r"(?<=\/image)[0-9]+(?=.)"
    files = sorted(files, key = lambda file: int(re.search(reg_exp, file).group()))
    for file in files:
        img = SudokuImage(file)
        print(img.shortened_filename)
        try:
            board, board_binary, board_size = img.find_board_location()
            multi_image_show_matplotlib([board, board_binary], 2, 1)
        except:
            # Doesn't raise an Exception in order to debug bad images faster.
            print(f"{img.shortened_filename}-problematic")
            continue

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
    #return [lableded_cell for img, data in pairs for lableded_cell in label_cells(return_cells(img), read_dat(data))]

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

def rename_files(directory):
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
    #print(files)
    reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\" else r"(?<=\/image)[0-9]+(?=.)"
    #print(reg_exp, "\n", os.sep)
    try:
        files = sorted(files, key = lambda file: int(re.search(r"(?<=(\\|\/)image)[0-9]+(?=.)", file).group()))
    except AttributeError:
        raise AttributeError("Incompatible/misnamed file found in directory")
    
    img_number = 1
    dat_number = 1
    for file in files:
        if file.lower().endswith(".jpg"):
            os.rename(file, f"dataset\image{img_number}.jpg")
            img_number += 1
        elif file.lower().endswith(".dat"):
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

def main():
    #win = FileDialogWindow()
    #make_cells(win.filename)
    output_csv("input", "custom_combined", combined=True)
    #rename_files("dataset")



if __name__ == "__main__":
    main()