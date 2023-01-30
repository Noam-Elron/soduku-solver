import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import re
import os
from utils import predict, image_show_matplotlib, convert_dtype
import matplotlib.pyplot as plt

class FileDialogWindow:
    root = tk.Tk()
    root.title('Tkinter Open File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    def __init__(self):
        self.filename = None
        self.file_dialog()

    def file_dialog(self):
        

        def select_files():
            filetypes = (
                ('text files', '*.jpg *.png'),
                ('All files', '*.*')
            )

            self.filename = fd.askopenfilename(
                title='Open a file',
                initialdir='C:\Downloads',
                filetypes=filetypes)

            if self.filename is not None:
                self.root.destroy()


        # open button
        open_button = ttk.Button(
            self.root,
            text='Open a File',
            command=select_files
            
        )

        open_button.pack(expand=True)
        
        
        # run the application
        self.root.mainloop() 

class SudokuImage:
    def __init__(self, filename, pos_x = 500, pos_y = 50):
        self.filename = filename
        self.shortened_filename = re.search("(?<=\/)\w+(?=\.+(jpg|png|jpeg))", self.filename).group()

        self.pos_x = pos_x
        self.pos_y = pos_y
        
        self.img = cv.imread(self.filename)
        self.board_loc = None
        self.warped = None
        self.cells = None

    @staticmethod
    def image_preprocess(img_copy, blocksize, c):
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        img_copy = cv.adaptiveThreshold(img_copy,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize, c)
        img_copy = cv.bitwise_not(img_copy)
        return img_copy

    def find_board(self, blocksize, c):
        filtered_img = self.image_preprocess(self.img.copy(), blocksize, c)
        contours, hierarchy = cv.findContours(filtered_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Returns the list of contours in descending area by contourArea
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        #Contours return in this order: Top Right, Top Left, Bottom Left, Bottom Right
        # Loops over all the contours, who are already sorted by descending area, until it finds the biggest area that is also a quadrillateral
        for contour in contours:
            
            arclen = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.015 * arclen, True)
            if len(approx) == 4:
                self.board_loc = approx
                break 
            
        self.board_loc = self.reorder()
        #self.img = cv.drawContours(self.img, self.board_loc, -1, (0, 255, 0), 3)  

        
        self.warped = self.warp_perspective(self.board_loc)
        self.warped = cv.cvtColor(self.warped.copy(), cv.COLOR_BGR2GRAY)
        self.cells = self.split_image(self.warped)
        print(self.cells.shape)
        self.cells = self.cells.reshape(-1, 28, 28, 1)
        cell = self.cells[3]
        self.img = cell
        #self.output_image(cell)
        cell = cell.reshape(-1, 28, 28, 1)
        prediction_list, prediction = predict(255*cell)
        print(prediction_list)
        print(f'Prediction: {prediction}')
        print(cell.shape)
        image_show_matplotlib(cell[0])
        self.output_image(self.img)

    def warp_perspective(self, location):
        """Apply a perspective warp - a.k.a creating a new image with a warped perspective so that 
            the area were interested in (the outerbound sudoku board grid) is the only thing
            in the new image, but with a birds eye view of it"""
        #-------------------------- #Top Left#  #Bottom Left# #Bottom Right# #Top Right#
        top_left, top_right, bottom_left, bottom_right = location[0], location[1], location[2], location[3]
        size = 900
        source_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")
        destination_points = np.array([[0, 0], [size, 0], [0, size], [size, size]], dtype="float32")
        # Apply Perspective Transform Algorithm
        matrix = cv.getPerspectiveTransform(source_points, destination_points)
        result = cv.warpPerspective(self.img, matrix, (size, size))
        return result

    
    def split_image(self, image) -> list:
        """Takes a sudoku board and split it into 81 cells. 
        each cell contains an element of that board either given or an empty cell."""
        # Splits the image into 9 equal-size sub-arrays
        rows = np.vsplit(image,9)
        cells = []
        for row in rows:
            cols = np.hsplit(row,9)
            for cell in cols:
                # Resize cell to be 125x125 and divide every value by 255 to make sure no pixel exceeds 255.
                cell = cv.resize(cell, (28, 28))/255.0
                #cell = self.convert_dtype(cell, 0, 255, np.uint8)/255.0
                cells.append(cell)
        return np.array(cells)   


    def reorder(self):
        """Reorders contours so perspective warp wont rotate the image"""
        # Flatten the board_loc array because it had a useless "dimension"(extra brackets, array shape was (4,1,2) aka 4 2D arrays where each 2D array contained a single 1D array and each 1D array contained the points. So reshape into one single 2D array which holds 4 1D arrays)
        self.board_loc = self.board_loc.reshape(4, 2)
        # Convert to normal python list.
        self.board_loc = self.board_loc.tolist()
        
        sums = [np.sum(points) for points in self.board_loc]
        sums = np.array(sums)
        linked = list(zip(self.board_loc, sums))
        bottom_right = linked[np.argmax(sums)][0]
        top_left = linked[np.argmin(sums)][0]
        
        points_left = [points for points in self.board_loc if points != top_left and points != bottom_right]
        # Two points remaining are top right and bottom left. Whoever has lowest y value is at top right, whoever is left is bottom left
        top_right = points_left[0] if points_left[0][1] < points_left[1][1] else points_left[1]
        bottom_left = points_left[0] if points_left[0][1] > points_left[1][1] else points_left[1]

        return np.array([top_left, top_right, bottom_left, bottom_right])
    
    def output_cells(self, num_cells):
        path = os.path.join(os.getcwd(), "training_data")
        for i in range(num_cells):
            cell_path = os.path.join(path, f'{self.shortened_filename}-{i}.jpg')
            cell = cv.resize(self.cells[i], (28, 28))
            cv.imwrite(cell_path, 255*cell)

    def output_image(self, image):
        path = os.path.join(os.getcwd(), "images")
        cell_path = os.path.join(path, f'{self.shortened_filename}-test.jpg')
        cv.imwrite(cell_path, 255*image)

    def image_show(self, blocksize, c):
        self.find_board(blocksize, c)    
        cv.imshow(f"{self.filename}", self.img)  
        cv.moveWindow(f"{self.filename}", self.pos_x, self.pos_y)
        cv.waitKey(0)

def main():
    win = FileDialogWindow()
    blocksize, c = 23, 7
    image = SudokuImage(win.filename)    
    image.image_show(blocksize, c)

    


    
if __name__ == "__main__":
    main()

    