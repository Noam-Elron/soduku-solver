import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import re
import os
from utils import predict_all, prediction_show_matplotlib, multi_image_show_matplotlib, image_show_matplotlib, convert_dtype
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
                initialdir='D:\Downloads\Creation\PythonCreations\soduku-solver\images',
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
        self.cells = None

    @staticmethod
    def image_preprocess(img_copy, blocksize, c):
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        img_copy = cv.adaptiveThreshold(img_copy,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize, c)
        return img_copy

    @staticmethod
    def image_preprocess_alternative(img_copy):
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)  
        img_copy = cv.GaussianBlur(img_copy, (5, 5), 1)  
        img_copy = cv.adaptiveThreshold(img_copy, 255, 1, 1, 11, 2)  
        img_copy = cv.bitwise_not(img_copy)
        return img_copy

    def find_board(self, blocksize, c):
        self.img = cv.resize(self.img, (900, 900))
        
        image_binary = self.image_preprocess(self.img.copy(), blocksize, c)
        img_for_contour = cv.bitwise_not(image_binary)
        contours, _ = cv.findContours(img_for_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Returns the list of contours in descending area by contourArea
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        #Contours return in this order: Top Right, Top Left, Bottom Left, Bottom Right
        # Loops over all the contours, who are already sorted by descending area, until it finds the biggest area that is also a quadrillateral
        for contour in contours:
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.1*peri, True)
            area = cv.contourArea(contour)
            if len(approx) == 4:
                board_loc = approx
                break 

        #self.img = cv.drawContours(self.img.copy(), board_loc, -1, (0, 255, 0), 3)  
        board_loc = self.reorder(board_loc)
        imgx, imgy, imgw, imgh = cv.boundingRect(board_loc)
        board_size = max(imgw, imgh)
        #print(f"contourArea: {area}, Board Size: {board_size}")

        warped, warped_binary = self.warp_perspective(self.img.copy(), image_binary, board_loc, board_size)

        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
        cells = self.get_cells(warped, board_size)
        cells_binary = self.get_cells(warped_binary, board_size)
        grid_string = self.create_board(cells_binary)
        #multi_image_show_matplotlib([warped, warped_binary], 2, 1)

        #[print(f"Prediction at Cell #{prediction[1]} is: {prediction[0]}") for prediction in predictions]
        #imgs = [*cells[0:9], *cells_binary[0:9]]
        #imgs = np.asarray(imgs)
    
        #multi_image_show_matplotlib(imgs, 18, 9)
        return grid_string

    def create_board(self, cells):
            digits_data = self.extract_numbers(cells)
            digits = [data[0] for data in digits_data]
            pos_data = [data[1] for data in digits_data]
            digits = np.reshape(digits, (-1, 28, 28, 1))
            #print(digits.shape)
            #prediction_show_matplotlib(digits)

            predictions = predict_all(digits, pos_data)
            grid = np.zeros(81, int)
            for prediction, position in predictions:
                grid[position] = prediction
            #print(grid)
            grid_stringified = ''.join(map(str, grid))
            return grid_stringified

    @staticmethod
    def pad_image_even(image, pixels: int, color: int):
        image = cv.copyMakeBorder(image, pixels, pixels, pixels, pixels, cv.BORDER_CONSTANT, color)
        return image

    def extract_numbers(self, cells):
        digits = []
        for i, cell in enumerate(cells):
            cell = self.pad_image_even(cell, 5, 0)
            #print(cell.shape)
            #image_show_matplotlib(cell, f"Cell #{i}")
            contours, _ = cv.findContours(cell, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                imgx, imgy, img_width, img_height = cv.boundingRect(contour)
                cell_drawn = cell.copy()
                cell_drawn = cv.drawContours(cell_drawn, contour, -1, 255, 3)
                # Img_height needs to be atleast 0.75 
                #print(f"Height: {img_width}, Width: {img_height}, Cell Shape: {cell.shape}, Cell #{i}")
                if not(img_height>3*img_width) and img_width > 0.3*img_height and img_width*img_height > (0.075*(cell.shape[0] * cell.shape[1])) and img_width*img_height < (0.6*(cell.shape[0] * cell.shape[1])):
                    digit = cell[imgy:imgy+img_height, imgx:imgx+img_width]
                    digit = cv.resize(digit, (28,28))
                    digit = self.pad_image_even(digit, 5, 0)
                    digit = cv.resize(digit, (28,28))
                    #cv.rectangle(cell, (imgx,imgy), (imgx+img_width,imgy+img_height), (0, 255, 0), 3)
                    #image_show_matplotlib(digit, f"Cell #{i}")
                    digits.append((digit, i))
                    break
        return digits

    def get_cells(self, warped_image, board_size):
        cells = self.split_image(warped_image, board_size) 
        cells = np.asarray(cells)
        #cells = [cv.resize(cells[i], (28,28)) for i in range(len(cells))]  
        #cells = np.reshape(cells, (-1, 28, 28, 1))
        cells = cv.bitwise_not(cells)
        return cells

    def warp_perspective(self, image, image_binary, location, size):
        """Apply a perspective warp - a.k.a creating a new image with a warped perspective so that 
            the area were interested in (the outerbound sudoku board grid) is the only thing
            in the new image, but with a birds eye view of it"""
        #-------------------------- #Top Left#  #Bottom Left# #Bottom Right# #Top Right#
        top_left, top_right, bottom_left, bottom_right = location[0], location[1], location[2], location[3]

        source_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")
        destination_points = np.array([[0, 0], [size, 0], [0, size], [size, size]], dtype="float32")
        # Apply Perspective Transform Algorithm
        matrix = cv.getPerspectiveTransform(source_points, destination_points)
        result = cv.warpPerspective(image, matrix, (size, size))
        result_binary = cv.warpPerspective(image_binary, matrix, (size, size))
        return result, result_binary

    
    def split_image(self, warped_image, board_size):
        cells = []
        cell_width, cell_height = board_size//9, board_size//9
        for y in range(9):
            for x in range(9):
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = x*cell_width, y*cell_height, (x + 1)*cell_width, (y + 1)*cell_height

                cell = warped_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                #cell = cv.resize(cell, (28,28))        
                cells.append(cell)

        return cells   

    def reorder(self, board_location):
        """Reorders contours so perspective warp wont rotate the image"""
        # Flatten the board_loc array because it had a useless "dimension"(extra brackets, array shape was (4,1,2) aka 4 2D arrays where each 2D array contained a single 1D array and each 1D array contained the points. So reshape into one single 2D array which holds 4 1D arrays)
        board_location = board_location.reshape(4, 2)
        # Convert to normal python list.
        board_location = board_location.tolist()
        
        sums = [np.sum(points) for points in board_location]
        sums = np.array(sums)
        linked = list(zip(board_location, sums))
        bottom_right = linked[np.argmax(sums)][0]
        top_left = linked[np.argmin(sums)][0]
        
        points_left = [points for points in board_location if points != top_left and points != bottom_right]
        # Two points remaining are top right and bottom left. Whoever has lowest y value is at top right, whoever is left is bottom left
        top_right = points_left[0] if points_left[0][1] < points_left[1][1] else points_left[1]
        bottom_left = points_left[0] if points_left[0][1] > points_left[1][1] else points_left[1]

        return np.array([top_left, top_right, bottom_left, bottom_right])
    
    def output_image(self, image):
        path = os.path.join(os.getcwd(), "images")
        cell_path = os.path.join(path, f'{self.shortened_filename}-output.jpg')
        cv.imwrite(cell_path, image)

    def image_show(self, blocksize, c):
        self.find_board(blocksize, c)    
        cv.imshow(f"{self.filename}", self.img)  
        cv.moveWindow(f"{self.filename}", self.pos_x, self.pos_y)
        cv.waitKey(0)

def main():
    win = FileDialogWindow()
    blocksize, c = 23, 7
    image = SudokuImage(win.filename)    
    grid = image.find_board(blocksize, c)

    


    
if __name__ == "__main__":
    main()

