import os
import glob
import time
import re

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import cv2 as cv

def pad_image(image, pixels: int, color: int):
    # args are src, top padding, bottom padding, left padding, right padding, border type, border value/color if type is constant.
    image = cv.copyMakeBorder(image, pixels, pixels, pixels, pixels, cv.BORDER_CONSTANT, color)
    return image

def return_dataset_images(directory: str, reverse: bool = False):
    """Returns all images from the dataset, expects images to be in the format image#number.file_extension where file_extension is either jpg,jpeg or png.

    Args:
        directory (str): Images Directory to search
        reverse (bool, optional): Flag to determine if images are returned in ascending #number order or descending. Defaults to False which is Ascending.

    Returns:
        filepaths: List of paths to each image in the directory.
    """
    cur_dir = os.getcwd()
    path = os.path.join(cur_dir, directory)
    # Normalize filepath to work for both windows and linux
    path = os.path.normcase(path) 
    files = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.jpeg")) + glob.glob(os.path.join(path, "*.png"))
    # Sorting the files based on number.
    reg_exp = r"(?<=\\image)[0-9]+(?=.)" if os.sep == "\\"  else r"(?<=\/image)[0-9]+(?=.)"
    files = sorted(files, key = lambda file: int(re.search(reg_exp, file).group()), reverse=reverse)
    return files





























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