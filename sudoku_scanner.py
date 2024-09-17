import cv2 as cv
import numpy as np
import re
import os
from fileprompter import FileDialogWindow
from utils import pad_image, predict_all, prediction_show_matplotlib, multi_image_show_matplotlib, image_show_matplotlib, convert_dtype
import matplotlib.pyplot as plt


class SudokuImage:
    def __init__(self, filename, blocksize = 23, c = 7):
        self.filename = filename
        self.shortened_filename = re.search(r"(?<=(\/|\\))\w+(?=\.+(jpg|png|jpeg))", self.filename).group()
        
        self.img = cv.imread(self.filename)
        self.__cells = None

        self.blocksize = blocksize
        self.c = c

    def return_board(self):
        cells = self.return_all_cells(binary=True)
        #multi_image_show_matplotlib(self.__cells, 20, 5)

        digits, digit_positions = self.extract_digits(cells)

        #prediction_show_matplotlib(digits)
        grid = self.predict_board(digits, digit_positions)
        grid_stringified = ''.join(map(str, grid))

        #imgs = [*cells[0:9], *cells_binary[0:9]]
        #imgs = np.asarray(imgs)    
        #multi_image_show_matplotlib(imgs, 18, 9)

        return grid_stringified

    def return_all_cells(self, binary=True):
        board, board_binary, board_size = self.find_board_location()
        #multi_image_show_matplotlib([board, board_binary], 2, 1)
        if binary==True:
            self.__cells = self.get_cells(board_binary, board_size)
        else:
            self.__cells = self.get_cells(board, board_size)
        return self.__cells
        
    def find_board_location(self):
        """
        Finds board's grid area and returns a warped image of the AOI(Area of Interest)

        Parameters:
            None

        Returns:
            Returns:
                warped: Image, warped image of the AOI
                warped_binary: warped binary image of the AOI
                board_size: int, literally just the board size...
        
        Raises:
            Nothing
        """

        self.img = cv.resize(self.img, (900, 900))
        
        image_binary = self.image_preprocess(self.img.copy())
        img_for_contour = cv.bitwise_not(image_binary)
        contours, _ = cv.findContours(img_for_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Returns the list of contours in descending area by contourArea
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        # Loops over all the contours, who are already sorted by descending area, until it finds the biggest area that is also a quadrillateral
        for contour in contours:
            #self.show_contour(self.img, contour)
            peri = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.1*peri, True)
            if len(approx) == 4:
                board_loc = approx
                break 
        try:
            board_loc = self.reorder(board_loc)
        except UnboundLocalError:
            #raise Exception("Images bad, no contour with 4 vertexes was found A.K.A unable to find sudoku boundary, try again with different image")
            return
        imgx, imgy, imgw, imgh = cv.boundingRect(board_loc)
        board_size = max(imgw, imgh)

        warped, warped_binary = self.warp_perspective(self.img.copy(), image_binary, board_loc, board_size)

        warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

        return warped, warped_binary, board_size


    def extract_digits(self, cells):
        """

        """
        digits = []
        positions = []
        for i, cell in enumerate(cells):
            cell = pad_image(cell, 5, 0)
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
                    digit = cv.resize(digit, (18,18))
                    digit = pad_image(digit, 5, 0)
                    #digit = cv.resize(digit, (28,28))
                    #cv.rectangle(cell, (imgx,imgy), (imgx+img_width,imgy+img_height), (0, 255, 0), 3)
                    #image_show_matplotlib(digit, f"Cell #{i}")
                    digits.append(digit)
                    positions.append(i)
                    break

        digits = np.reshape(digits, (-1, 28, 28, 1))
        return digits, positions

    def get_cells(self, warped_image, board_size):
        cells = self.split_image(warped_image, board_size) 
        cells = np.asarray(cells)
        #cells = [cv.resize(cells[i], (28,28)) for i in range(len(cells))]  
        #cells = np.reshape(cells, (-1, 28, 28, 1))
        cells = cv.bitwise_not(cells)
        return cells

    def warp_perspective(self, image, image_binary, location, size):
        """Apply a perspective warp - a.k.a creating a new image with a warped perspective so that 
            the area were interested in(ROI - Region of interest - the outerbound sudoku board grid) is the only thing
            in the new image, with a birds eye view"""
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
        """ 
        Returns all the cells of the sudoku board
        Takes a warped image w/ the size of the board as arguments
        """
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
    
    def predict_board(self, digits, digit_positions):
        predictions = predict_all(digits, digit_positions)
        grid = np.zeros(81, int)
        for prediction, position in predictions:
            grid[position] = prediction
        #print(grid)
        return grid

    def image_preprocess(self, img_copy):
        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        img_copy = cv.adaptiveThreshold(img_copy,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, self.blocksize, self.c)
        return img_copy

    def output_cells(self, num_cells, directory):
        path = os.path.join(os.getcwd(), f"{directory}")
        for i in range(num_cells):
            cell = cv.resize(self.__cells[i], (28, 28))
            cell_path = os.path.join(path, f'{self.shortened_filename}-{i}.jpg')
            cv.imwrite(cell_path, cell)

    def output_image(self, image):
        path = os.path.join(os.getcwd(), "images")
        cell_path = os.path.join(path, f'{self.shortened_filename}-output.jpg')
        cv.imwrite(cell_path, image)

    def image_show(self, pos_x = 500, pos_y = 50):
        self.find_board(self.blocksize, self.c)    
        cv.imshow(f"{self.filename}", self.img)  
        cv.moveWindow(f"{self.filename}", pos_x, pos_y)
        cv.waitKey(0)

    def show_contour(self, img, contour):
        copy = cv.drawContours(img.copy(), contour, -1, 255, 3)
        cv.imshow(f"{self.shortened_filename}-contour", copy)
        cv.waitKey(0)

def main():
    win = FileDialogWindow("dataset")
    image = SudokuImage(win.filename)    
    image.find_board_location
    grid = image.return_board()
    


    
if __name__ == "__main__":
    main()

