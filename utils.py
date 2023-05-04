import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import math
import cv2 as cv
import time

def image_show_matplotlib(image, title="Image"):
    fig = plt.figure()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def multi_image_show_matplotlib(images, num_images, num_in_row):
    """
    Show multiple images in same matplotlib figure.
    """
    rows = math.ceil(num_images/num_in_row)
    fig, axis = plt.subplots(nrows=rows, ncols=num_in_row, figsize=(12, 14))
    for i, ax in enumerate(axis.flat):
        ax.imshow(images[i], cmap='gray', interpolation='none')
        ax.set(title = f"Image #{i+1}")
    plt.show()

def predict(image):
    from tensorflow.keras.models import load_model
    model = load_model('models/attempt.h5')
    #print(model.summary())
    config = model.get_config() # Returns pretty much every information about your model
    print("Input: ", config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    prediction = model.predict(image)

    return prediction, prediction.argmax() 

def predict_all(images, positions):
    from tensorflow.keras.models import load_model
    model = load_model('models/combined_dataset2.h5')
    predictions = model.predict(images)
    predict_list = [[predictions[i].argmax(), positions[i]] for i, pred in enumerate(predictions)]
    return predict_list    
    

def prediction_show_matplotlib(cells):
    from tensorflow.keras.models import load_model
    model = load_model('models/combined_dataset2.h5')
    pred = model.predict(cells)
    fig, axis = plt.subplots(4, 4, figsize=(12, 14))
    for i, ax in enumerate(axis.flat):
        ax.imshow(cells[i], cmap='gray')
        ax.set(title = f"Predicted Number is {pred[i].argmax()}")
    plt.show()


def pad_image(image, pixels: int, color: int):
    image = cv.copyMakeBorder(image, pixels, pixels, pixels, pixels, cv.BORDER_CONSTANT, color)
    return image

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

def convert_dtype(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def s(img):
    w, h = img.shape[1], img.shape[0]
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(img_gray, 100, 400)

    blurred = cv.GaussianBlur(img_gray, (11, 11), 0)
    img_bw = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # cv.imwrite("img_out.png", img_out)

    t_start = time.monotonic()

    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_out = img.copy()
    # cv.drawContours(img_out, contours, -1, (0, 255, 0), 1)
    result_found = False
    for cntr in contours:
        imgx, imgy, imgw, imgh = cv.boundingRect(cntr)
        if imgw < w/5 or imgw < h/5 or imgw/imgh < 0.25 or imgw/imgh > 1.5:
            continue

        def normalize_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        # Approximate the contour and apply the perspective transform
        peri = cv.arcLength(cntr, True)
        frm = cv.approxPolyDP(cntr, 0.1*peri, True)
        if len(frm) != 4:
            continue

        # Converted image should fit into the original size
        board_size = max(imgw, imgh)
        if len(frm) != 4 or imgx + board_size >= w or imgy + board_size >= h:
            continue
        # Points should not be too close to each other (use euclidian distance)
        if cv.norm(frm[0][0] - frm[1][0], cv.NORM_L2) < 0.1*peri or \
            cv.norm(frm[2][0] - frm[1][0], cv.NORM_L2) < 0.1*peri or \
            cv.norm(frm[3][0] - frm[1][0], cv.NORM_L2) < 0.1*peri or \
            cv.norm(frm[3][0] - frm[2][0], cv.NORM_L2) < 0.1*peri:
            continue

        # Draw sudoku contour
        cv.line(img_out, frm[0][0], frm[1][0], (0, 200, 0), thickness=3)
        cv.line(img_out, frm[1][0], frm[2][0], (0, 200, 0), thickness=3)
        cv.line(img_out, frm[2][0], frm[3][0], (0, 200, 0), thickness=3)
        cv.line(img_out, frm[0][0], frm[3][0], (0, 200, 0), thickness=3)
        cv.drawContours(img_out, frm, -1, (0, 255, 255), 10)

        # Source and destination points for the perspective transform
        src_pts = normalize_points(frm.reshape((4, 2)))
        dst_pts = np.array([[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]], dtype=np.float32)
        t_matrix = cv.getPerspectiveTransform(src_pts, dst_pts)
        _, t_matrix_inv = cv.invert(t_matrix)

        # Convert images, colored and monochrome
        warped_disp = cv.warpPerspective(img, t_matrix, (board_size, board_size))
        warped_bw = cv.warpPerspective(img_bw, t_matrix, (board_size, board_size))

        # Sudoku board found, extract digits from the 9x9 grid
        images = []
        cell_w, cell_h = board_size//9, board_size//9
        for x in range(9):
            for y in range(9):
                x1, y1, x2, y2 = x*cell_w, y*cell_h, (x + 1)*cell_w, (y + 1)*cell_h
                cx, cy, w2, h2 = (x1 + x2)//2, (y1 + y2)//2, cell_w, cell_h
                # Find the contour of the digit
                crop = warped_bw[y1:y2, x1:x2]
                cntrs, _ = cv.findContours(crop, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                for dc in cntrs:
                    imgx2, imgy2, imgw2, imgh2 = cv.boundingRect(dc)
                    if 0.2 * w2 < imgw2  < 0.8 * w2 and 0.4 * h2 < imgh2 < 0.8 * h2:
                        cv.rectangle(warped_disp, (x1 + imgx2, y1 + imgy2), (x1 + imgx2 + imgw2, y1 + imgy2 + imgh2), (0, 255, 0), 1)
                        cv.imshow("s", warped_disp)
                        cv.waitKey(0)
                        digit_img = crop[imgy2:imgy2 + imgh2, imgx2:imgx2 + imgw2]
                        images.append(digit_img)
                        break
        return images

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
