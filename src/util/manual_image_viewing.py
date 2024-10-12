import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import cv2 as cv
from sudoku_logic.sudoku_scanner import SudokuImage
from .utils import return_dataset_images
from .imgplotting import image_show_matplotlib, multi_image_show_matplotlib
from matplotlib.widgets import Button, Slider

def look_at_dataset_warped(directory: str, reverse: bool = False):
    
    """
    Shows every image's warped perspective that is inside the given directory

    Parameters:
        directory(str): directory that contains images whose names start in image followed by a number

    Returns:
        Nothing
    
    Raises:
        Nothing
    """
    files = return_dataset_images(directory, reverse)
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

def configure_threshold(directory: str, reverse: bool = False):

    """
    Creates a window with multiple images

    Parameters:
        directory(str): directory that contains images whose names start in image followed by a number

    Returns:
        Nothing
    
    Raises:
        Nothing
    """
    files = return_dataset_images(directory, reverse)
    for file in files:
        img = SudokuImage(file)
        _, warped_board_binary, _ = img.find_board_location()

        try:
            fig, axes = plt.subplots(ncols=2)
            axes_img1 = axes[0].imshow(img.img, cmap='gray', interpolation='none')
            axes_img2 = axes[1].imshow(warped_board_binary, cmap='gray', interpolation='none')
            axes_imgs = [axes_img1, axes_img2]
            plt.subplots_adjust(bottom=0.35)

            cfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            c_slider = Slider(ax=cfreq, label='c', valmin=0, valmax=40, valinit=7, valstep=1,)

            blocksizefreq = fig.add_axes([0.25, 0.3, 0.65, 0.03])
            bs_slider = Slider(ax=blocksizefreq, label='Blocksize', valmin=1, valmax=300, valinit=23, valstep=[i for i in range(1, 300, 2)],)
            axes[0].set_title(img.shortened_filename)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            axes[1].set_title(img.shortened_filename + " Warped Binary")
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            c_slider.on_changed(lambda val: update_c(img, axes_imgs, val))
            bs_slider.on_changed(lambda val: update_bs(img, axes_imgs, val))

            plt.show()
        except Exception as e:
            # Doesn't raise an Exception in order to debug bad images faster.
            raise e
            print(f"{img.shortened_filename}-problematic")
            continue
            
def update_c(img, axes_imgs, val):
    img.c = val
    axes_imgs[0].set_data(img.binarize_image(img.img.copy()))
    axes_imgs[1].set_data(img.find_board_location()[1])
def update_bs(img, axes_imgs, val):
    img.blocksize = val
    axes_imgs[0].set_data(img.binarize_image(img.img.copy()))
    axes_imgs[1].set_data(img.find_board_location()[1])


# register the update function with each slider

