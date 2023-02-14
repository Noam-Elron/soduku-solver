from fileprompter import FileDialogWindow
from sudoku_scanner import SudokuImage
import cv2 as cv
import numpy as np
import os
from utils import multi_image_show_matplotlib, pad_image
import glob 
import re
from typing import List, Tuple

def get_all_data_pairs() -> List[Tuple[str, str]]:
    """
    Returns a zipped list of tuples containing matching pairs of (img, data).
    """
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, "dataset")
    # Normalize filepath to work for both windows and linux
    path = os.path.normcase(path) 
    imgs = []
    data = []
    files = glob.glob(os.path.join(path, "*"))
    reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\"  else r"(?<=\/image)[0-9]+(?=.)"
    files = sorted(files, key = lambda file: int(re.search(reg_exp, file).group()))
    for file in files:
        if file.lower().endswith(".jpg"):
            imgs.append(file)
        elif file.lower().endswith(".dat"):
            data.append(file)
    
    pairs = list(zip(imgs, data))
    return pairs

def return_cells(filename: str):
    """ Takes a filename and returns a flattened array of all"""
    img = SudokuImage(filename)
    cells = img.return_all_cells()
    print(cells.shape)
    # Resize all cells to be a 28x28 image to be uniform with 
    #multi_image_show_matplotlib(cells, 20, 4)
    cells = [cv.resize(cells[i], (18,18)) for i in range(len(cells))]  
    cells = [pad_image(cell, 5, 0) for cell in cells]
    #multi_image_show_matplotlib(cells, 20, 4)
    cells = np.reshape(cells, (-1, 28*28)) 
    return cells
"""
TODO
1) Convert all dat files to a better format/read the data(digits) from the dat files #DONE
2) Create rename_cells function which takes the data and renames the cell into image##-digit
3) Convert array of total_cells in output_csv function to csv file.
4) Test out with, say 3 images to not create a cluttered csv file.
"""

def read_dat(data: str) -> List[str]:
    """
    Returns all the digits belonging to the associated(same filename) sudoku board image. 
    Receives file path for a .dat file that contains the mentioned digits. 
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

def rename_cells(cells, data):
    pass

def output_csv(pairs: List[Tuple[str, str]]):

    for (img, data) in pairs:
        cells = return_cells(img)
        labels = read_dat(data)

        # Need to create pandas dataframe where column 0 is Labels, and every column afterwards is a single pixel value. Maybe make column 0 filename?

    # TODO : Convert cells to csv
    # Writeout csv after finishing.

def look_at_dataset():
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, "good_dataset")
    # Normalize filepath to work for both windows and linux
    path = os.path.normcase(path) 
    import glob
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
            print(f"{img.shortened_filename}-problematic")
            continue

def main():
    #win = FileDialogWindow()
    #make_cells(win.filename)
    pairs = get_all_data_pairs()
    pair = pairs[0]
    print(pair)
    img, data = pair[0], pair[1]
    #read_dat(data)
    return_cells(img)



if __name__ == "__main__":
    main()