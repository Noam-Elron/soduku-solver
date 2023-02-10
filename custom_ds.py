from fileprompter import FileDialogWindow
from sudoku_scanner import SudokuImage
import cv2 as cv
import numpy as np
import os
from utils import multi_image_show_matplotlib
import glob 
import re


def get_all_data_pairs():
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, "dataset")
    imgs = []
    data = []
    files = glob.glob(os.path.join(path, "*"))
    files = sorted(files, key = lambda file: int(re.search(r"(?<=\\image)[0-9]+(?=.)", file).group()))
    for file in files:
        if file.lower().endswith(".jpg"):
            imgs.append(file)
        elif file.lower().endswith(".dat"):
            data.append(file)
    pairs = list(zip(imgs, data))
    return pairs

def return_cells(filename):
    img = SudokuImage(filename)
    cells = img.return_all_cells()
    cells = [cv.resize(cells[i], (28,28)) for i in range(len(cells))]  
    cells = np.reshape(cells, (-1, 28*28))
    #multi_image_show_matplotlib(cells, 20, 4)
"""
TODO
1) Convert all dat files to a better format/learn how to read dat files
2) Create rename_cells function which takes the data and renames the cell into image##-digit
3) Convert array of total_cells in output_csv function to csv file.
4) Test out with, say 3 images to not create a cluttered csv file.
"""
def rename_cells(cells, data):
    pass

def output_csv():
    total_cells = []
    pairs = get_all_data_pairs()
    for (img, data) in pairs:
        cells = return_cells(img)
        cells = rename_cells(cells, data)
        total_cells.append(*cells)

    # TODO : Convert cells to csv
    # Writeout csv after finishing.

def main():
    #win = FileDialogWindow()
    #make_cells(win.filename)
    

if __name__ == "__main__":
    main()