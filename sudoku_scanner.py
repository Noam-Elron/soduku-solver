import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import imutils

class MainWindow:
    root = tk.Tk()
    root.title('Tkinter Open File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    def __init__(self):
        self.filenames = None

    # create the root window
    def file_dialog(self):
        

        def select_files():
            filetypes = (
                ('text files', '*.jpg *.png'),
                ('All files', '*.*')
            )

            self.filenames = fd.askopenfilename(
                title='Open a file',
                initialdir='C:\Downloads',
                filetypes=filetypes)

            if self.filenames is not None:
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

class Image:
    def __init__(self, filename, pos_x, pos_y):
        self.filename = filename
        self.pos_x = pos_x
        self.pos_y = pos_y
        
        self.img = cv.imread(self.filename)
        self.board_loc = None
        self.warped = None

    @staticmethod
    def image_preprocess(img_copy, blocksize, c):

        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        img_copy = cv.adaptiveThreshold(img_copy,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize, c)
        img_copy = cv.bitwise_not(img_copy)
        return img_copy

    def find_board(self, blocksize, c):
        filtered_img = self.image_preprocess(self.img.copy(), blocksize, c)
        contours, hierarchy = cv.findContours(filtered_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Returns the list of contours in descending area by contourArea
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        #Contours return in this order: Top Right, Top Left, Bottom Left, Bottom Right
        # Loops over all the contours, who are already sorted by descending area, until it finds the biggest area that is also a quadrillateral
        for contour in contours:
            approx = cv.approxPolyDP(contour, 15, True)
            if len(approx) == 4:
                self.board_loc = approx
                break

        #print(self.board_loc)
        self.board_loc = self.reorder()
        self.warped = self.get_perspective(self.board_loc)
        #self.warped = cv.cvtColor(self.warped, cv.COLOR_BGR2GRAY)
        self.img = self.warped

        
        squares = self.split_image(self.img)
        # Convert the squares array into a numpy ndarray to be able perform operations on it/just generally be faster.
        squares = np.array(squares)
        #squares = squares.reshape(-1, 125, 125, 1)
        #self.img = squares[80]
        
        
    def reorder(self):
        """Reorders contours so perspective warp wont rotate the image"""
        # Flatten the board_loc array because it had a useless "dimension"(extra brackets, array shape was (4,1,2) aka 4 2D arrays where each 2D array contained a single 1D array and each 1D array contained the points. So reshape into one single 2D array which holds 4 1D arrays)
        #print(self.board_loc)
        self.board_loc = self.board_loc.reshape(4, 2)
        self.board_loc = self.board_loc.tolist()
        print(self.board_loc)
        print("--------------------------------------------")
        
        
        sums = [np.sum(points) for points in self.board_loc]
        sums = np.array(sums)
        linked = zip(self.board_loc, sums)
        linked = list(linked)
        print(linked)
        bottom_right = linked[np.argmax(sums)][0]
        top_left = linked[np.argmin(sums)][0]
        
        #print(sums)
        #print(type(list(linked)[0][0]))
        points_left = [points for points in self.board_loc if points != top_left and points != bottom_right]
        print(points_left)
        # Two points remaining are top right and bottom left. Whoever has lowest y value is at top right, whoever is left is bottom left
        top_right = points_left[0] if points_left[0][1] < points_left[1][1] else points_left[1]
        bottom_left = points_left[0] if points_left[0][1] > points_left[1][1] else points_left[1]
        #print([top_left, bottom_left, bottom_right, top_right])
        return np.array([top_left, bottom_left, bottom_right, top_right])

        

    def get_perspective(self, location, height = 900, width = 900):
        """Apply a perspective warp - a.k.a creating a new image with a warped perspective so that 
            the area were interested in (the outerbound sudoku board grid) is the only thing
            in the new image, but with a birds eye view of it"""
        #-------------------------- #Top Left#  #Bottom Left# #Bottom Right# #Top Right#
        source_points = np.float32([location[0], location[1], location[2], location[3]])
        destination_points = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
        # Apply Perspective Transform Algorithm
        matrix = cv.getPerspectiveTransform(source_points, destination_points)
        result = cv.warpPerspective(self.img, matrix, (width, height))
        return result

    def split_image(self, image):
        """Takes a sudoku board and split it into 81 cells. 
        each cell contains an element of that board either given or an empty cell."""
        # Splits the image into 9 equal-size sub-arrays
        rows = np.vsplit(image,9)
        cells = []
        for row in rows:
            cols = np.hsplit(row,9)
            for cell in cols:
                # Resize cell to be 125x125 and divide every value by 255 to make sure no pixel exceeds 255.
                cell = cv.resize(cell, (125, 125))/255.0
                cells.append(cell)
        return cells   

    def image_show(self, blocksize, c):
        self.find_board(blocksize, c)
        cv.imshow(f"{self.filename}", self.img)      
        cv.moveWindow(f"{self.filename}", self.pos_x, self.pos_y)
        cv.waitKey(0)
        

def main():
    win = MainWindow()
    win.file_dialog()
    blocksize, c = 5, 3
    image = Image(win.filenames, 200, 200)    
    image.image_show(blocksize, c)

    


    
if __name__ == "__main__":
    main()

    