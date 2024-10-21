import cv2 as cv
import numpy as np
import re
import os
from util.fileprompter import FileDialogWindow
from util.imgplotting import multi_image_show_matplotlib, image_show_matplotlib
from util.predict import predict_all, prediction_show_matplotlib


class SudokuImage:
    def __init__(self, filename, blocksize = 53, c = 7):
        self.filename = filename
        self.shortened_filename = re.search(r"(?<=(\/|\\))\w+(?=\.+(jpg|png|jpeg))", self.filename).group()
        
        self.img = cv.resize(cv.imread(self.filename), (900, 900))
        
        self.blocksize = blocksize
        self.c = c

    def return_board(self, debug=False):
        """Finds all the sudoku cells in the image, "extracts" the digits images and their positions into respective arrays and then
        feeds the information into predict_board to receive all the actual numerical digits in string form. 

        Args:
            debug (bool, optional): Boolean clause to determine if to display results in debugging mode. Defaults to False.

        Returns:
            str: String format of actual board
        """
        
        cells = self.return_all_cells(debug, binary=True)
        
        digits, digit_positions = self.extract_digits(cells)
        

        if debug:
            print(digits.shape)
            multi_image_show_matplotlib(cells, len(cells), 10)
            multi_image_show_matplotlib(digits, len(digits), 10)
            prediction_show_matplotlib(digits)

        #grid = self.predict_board(digits, digit_positions)
        #grid_stringified = ''.join(map(str, grid))

        #imgs = [*cells[0:9], *cells_binary[0:9]]
        #imgs = np.asarray(imgs)    
        #multi_image_show_matplotlib(imgs, 18, 9)

        #return grid_stringified

    def return_all_cells(self, debug=False, binary=True):
        """
        Returns an array of images of all the cells in the board

        Args:
            binary (bool, optional): Flag that determines if returns cells will be binarized. Defaults to True.

        Returns:
            List[images]: List of images of cells
        """
        warped_board, warped_board_binary, board_size = self.find_board_location(debug)
        cells = []
        if binary==True:
            cells = self.split_image_to_cells(warped_board_binary, board_size)
        else:
            warped_board = cv.cvtColor(warped_board, cv.COLOR_BGR2GRAY)
            cells = self.split_image_to_cells(warped_board, board_size)
        return cells
        
    def find_board_location(self, debug=False):
        """
        Finds board's grid area and returns a warped image of the AOI(Area of Interest)

        Parameters:
            debug (bool, optional): Boolean clause to determine if to display results in debugging mode. Defaults to False.

        Returns:
            warped: Image, warped image of the AOI
            warped_binary: warped binary image of the AOI
            board_size: int, literally just the board size...
        
        Raises:
            Nothing
        """

        # TODO Decide if i should perform other preprocessing steps in binarize_image
        image_binary = self.binarize_image(self.img.copy())

        # Deciding to make background black and foreground/objects white for convenience and consistency
        img_for_contour = cv.bitwise_not(image_binary)
        # Finds all external contours and approximates them into semi simple shapes/removes redundant points.
        contours, _ = cv.findContours(img_for_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Returns the list of contours in descending area by contourArea
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        # Loops over all the contours, who are already sorted by descending area, until it finds the biggest area that is also a quadrilateral
        for contour in contours:
            perimeter_len = cv.arcLength(contour, True)
            # Using the perimeter of the contour found we can approximate its shape, removing any small inconsistencies 
            # less than epsilon, leaving us hopefully with a perfect quadrilateral. 10% of perimeter length seems to be a good estimation
            # To remove inconsistencies while maintaining general shape of contour
            # Further experimentation with epsilon value may yield better results.
            approx = cv.approxPolyDP(contour, 0.1*perimeter_len, True)
            
            #print(f"{self.shortened_filename}, Contour approx points: {len(approx)}, Contour area: {cv.contourArea(contour)}")

            if len(approx) == 4 and cv.contourArea(contour) > 5000:
                board_loc = approx
                break 
        try:
            board_loc_original = board_loc # needed for properly drawing around the board using polylines function down below for debugging
            board_loc = self.reorder(board_loc)
            
        except UnboundLocalError:
            raise Exception("Image is bad, no contour with 4 vertexes was found or area was less than 5000, A.K.A unable to find sudoku boundary, try again with different image")
        
        # Returns top-left x,y coordinates of rectangle along with width and height.
        imgx, imgy, imgw, imgh = cv.boundingRect(board_loc)
        board_size = max(imgw, imgh)

        warped, warped_binary = self.warp_perspective(self.img.copy(), image_binary, board_loc, board_size)
        
        if debug:
            cv.polylines(self.img, np.int32([board_loc_original]), True, 255, 5) #draw board square onto original image.
            multi_image_show_matplotlib([self.img, cv.cvtColor(self.img, cv.COLOR_BGR2GRAY), image_binary, img_for_contour, warped, warped_binary], 6, 2)

        return warped, warped_binary, board_size
    
    def warp_perspective(self, image, image_binary, location, size):
        """
        Apply a perspective warp - a.k.a creating a new image with a warped perspective so that 
        the area we're interested in(ROI - Region of interest - the outerbound sudoku board grid) is the only thing
        in the new image, with a birds eye view
            

        Args:
            image: Original sudoku image
            image_binary: Binary version of sudoku image
            location: Location of the sudoku board corners in the source image
            size: Size of output image
        Returns:
            SudokuBoard, BinarySudokuBoard Images
        """
        #-------------------------- #Top Left#  #Bottom Left# #Bottom Right# #Top Right#
        top_left, top_right, bottom_left, bottom_right = location[0], location[1], location[2], location[3]

        source_points = np.array([top_left, top_right, bottom_left, bottom_right], dtype="float32")
        destination_points = np.array([[0, 0], [size, 0], [0, size], [size, size]], dtype="float32")
        # Apply Perspective Transform Algorithm
        # Creates an appropriate transformation matrix using the source points s.t that applying the matrix on the image yields the corresponding destination points for those points.
        matrix = cv.getPerspectiveTransform(source_points, destination_points)
        # Applies the transformation matrix on the original image.
        result = cv.warpPerspective(image, matrix, (size, size))
        result_binary = cv.warpPerspective(image_binary, matrix, (size, size))
        return result, result_binary
    

    def split_image_to_cells(self, warped_image, board_size):
        """Returns all the cells of the sudoku board
        Args:
            warped_image (Image): Warped image of the sudoku board
            board_size (int): Area of the board

        Returns:
            ndarray[Cells]: Numpy arrays of the cells of the sudoku board.
        """
        cells = []
        cell_width, cell_height = board_size//9, board_size//9
        
        for row in range(9):
            for column in range(9):
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = column*cell_width, row*cell_height, (column + 1)*cell_width, (row + 1)*cell_height

                # First dimension corresponds to rows(y), second to columns(x)
                cell = warped_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                cells.append(cell)

        cells = cv.bitwise_not(cells)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
        for i in range(len(cells)):
            cells[i] = cv.morphologyEx(cells[i], cv.MORPH_OPEN, kernel)
        
        return np.asarray(cells)

    def extract_digits(self, cells):
        """Extracts digits from sudoku cells

        Args:
            cells (Image): Cells of the sudoku board

        Returns:
            List[Images]: List of images of the extracted digits
        """
        digits = []
        positions = []
        for i, cell in enumerate(cells):
            cell = cv.copyMakeBorder(cell, 5, 5, 5, 5, cv.BORDER_CONSTANT, 0)
            
            contours, _ = cv.findContours(cell, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                imgx, imgy, img_width, img_height = cv.boundingRect(contour)
                # Img_height needs to be atleast 0.75 
                #print(f"Height: {img_width}, Width: {img_height}, Cell Shape: {cell.shape}, Cell #{i}")
                #print(f"not img_height>3*img_width: {not(img_height>3*img_width)}, img_width > 0.3*img_height: {img_width > 0.3*img_height}, img_width*img_height > (0.05*(cell.shape[0] * cell.shape[1])): {img_width*img_height > (0.05*(cell.shape[0] * cell.shape[1]))}, img_width*img_height < (0.6*(cell.shape[0] * cell.shape[1]): {img_width*img_height < (0.6*(cell.shape[0] * cell.shape[1]))}")
                if not(img_height>3*img_width) and img_width > 0.3*img_height and img_width*img_height > (0.05*(cell.shape[0] * cell.shape[1])) and img_width*img_height < (0.6*(cell.shape[0] * cell.shape[1])):
                    digit = cell[imgy:imgy+img_height, imgx:imgx+img_width]
                    digit = cv.resize(digit, (18,18))
                    digit = cv.copyMakeBorder(digit, 5, 5, 5, 5, cv.BORDER_CONSTANT, 0)
                    #cv.rectangle(cell, (imgx,imgy), (imgx+img_width,imgy+img_height), (0, 255, 0), 3)
                    #image_show_matplotlib(digit, f"Cell #{i}")
                    digits.append(digit)
                    positions.append(i)
                    break

        digits = np.reshape(digits, (-1, 28, 28, 1))
        return digits, positions


    def reorder(self, board_location):
        """Rearranges the points in board location to the format (top_left, top_right, bottom_left, bottom_right) so perspective warp wont rotate the image

        Args:
            board_location ndarray/image: Image of sudoku board extracted from original sudoku image.

        Returns:
            ndarray: Rearranged numpy array to specified format
        """
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
    
    def binarize_image(self, img_copy):
        """
        Given an image, converts it to GrayScale and then Binarizes it using Adaptive Thresholding.  

        Parameters:
            img_copy Image: Copy of image to binarize(to avoid side-effects), if not caring/wanting side-effects image itself may be passed.

        Returns:
            Image: Binarized version of image
             
        Raises:
            Nothing
        """

        img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
        img_copy = cv.adaptiveThreshold(img_copy,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, self.blocksize, self.c)
        return img_copy














    def extract_digits2(self):
        """
        Gets Cells based on contours, havent thought of way yet to find positions of cells though. So far failed attempt as unneccesary noise found...
        """
        
        warped_board, warped_board_binary, board_size = self.find_board_location()

        # Look at all contours in warped board, with the entire hierarchy tree.
        contours, hierarchies = cv.findContours(warped_board_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # For some reason hierarchies array is given wrapped inside an unnecessary dimension
        hierarchies = hierarchies[0]

        # Cant store numpy arrays as keys so need to use index as key.
        levels = list(map(lambda contour_index: self.get_hierarchy_level(hierarchies, contour_index), range(len(contours))))

        # Mapping each contour to its hierarchy level, going to use this to remove contours whose hierarchy level is 0, aka external contours aka the borders of the sudoku grid.
        # Mapping itself useful to sort based on hierarchy level and also filtering later
        # Hierarchy only useful for looking at the digits parent which will probably be the cell border theyre in
        contour_hierarchy_level_mapping = list(zip(contours, hierarchies, levels))
        contour_hierarchy_level_mapping = sorted(contour_hierarchy_level_mapping, key = lambda mapping: mapping[2])
        sorted_contours = [contour_mapping[0] for contour_mapping in contour_hierarchy_level_mapping]
        sorted_hierarchies = [hierarchy_mapping[1] for hierarchy_mapping in contour_hierarchy_level_mapping]

        # Removing all from list all contours whose hierarchy level is 0 aka external contours aka borders of sudoku grid. 
        filtered_mapping = list(filter(lambda mapping: mapping[2] > 0, contour_hierarchy_level_mapping))
        filtered_contours = [contour_mapping[0] for contour_mapping in filtered_mapping]
        sorted_hierarchies = [hierarchy_mapping[1] for hierarchy_mapping in filtered_mapping]
        
        # Useful for diagnostic image testing.
        color_mapping = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (0, 200, 200), 4: (255, 255, 0), 5: (127, 0 , 255), 6: (128, 128, 128)}

        # Applying boundingRect on filtered_contours returns an array where last two indices are img_width and img_height. Map is iterable thus can iterate over all arrays
        # obtained from operation and thus easily find each contours area
        contour_areas = [area_params[2]*area_params[3] for area_params in map(cv.boundingRect, filtered_contours) if area_params[2]*area_params[3] < 2500]
        
        # Reason why we wanted the contour areas is to find the mean value, we're assuming that 1. the area of an average digit is greater than the area of random dots of noise and 
        # 2. that theres not enough noise s.t that their accumulated sum affects the mean to such a degree that the results are truly skewed.
        mean = np.mean(contour_areas)
        """
        for i, contour in enumerate(sorted_contours):
            imgx, imgy, img_width, img_height = cv.boundingRect(contour)
            print(f"Area: {img_width*img_height}, Greater than half of mean: {img_width*img_height > 0.5*mean}, Mean: {mean}")
            warped_board = cv.drawContours(warped_board, sorted_contours, i, color_mapping[contour_hierarchy_level_mapping[i][2]], 5, cv.LINE_4) 
        image_show_matplotlib(warped_board)
        """
        for i, contour in enumerate(filtered_contours):
            imgx, imgy, img_width, img_height = cv.boundingRect(contour) # Cant simply replace with contour_areas[i] as imgx, imgy, img_width and img_height needed for extracting the digits later.
            
            if (0.5*mean) <= (img_width*img_height) <= (3*mean) and (img_height/img_width <= 3) and (img_width/img_height <= 3):
                contour_hierarchy = sorted_hierarchies[i]
                parentx, parenty, parent_width, parent_height = cv.boundingRect(contours[contour_hierarchy[3]])
                cv.rectangle(warped_board, (imgx,imgy), (imgx+img_width,imgy+img_height), (0, 255, 0), 3)
                #cv.rectangle(warped_board, (parentx,parenty), (parentx+parent_width,parenty+parent_height), (0, 0, 255), 3)
        
        image_show_matplotlib(warped_board)
        

    @staticmethod
    def get_hierarchy_level(hierarchies, contour_index):
        level = 0

        while hierarchies[contour_index][3] != -1:
            contour_index = hierarchies[contour_index, 3]
            level += 1
        return level










    def output_cells(self, cells, num_cells, directory):
        path = os.path.join(os.getcwd(), f"{directory}")
        for i in range(num_cells):
            cell = cv.resize(cells[i], (28, 28))
            cell_path = os.path.join(path, f'{self.shortened_filename}-{i}.jpg')
            cv.imwrite(cell_path, cell)



def main():
    win = FileDialogWindow("dataset")
    image = SudokuImage(win.filename)    
    image.find_board_location
    grid = image.return_board()
    


    
if __name__ == "__main__":
    main()

