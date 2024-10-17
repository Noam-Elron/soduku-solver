import matplotlib.pyplot as plt
import cv2 as cv
from sudoku_logic.sudoku_scanner import SudokuImage
from .imgplotting import image_show_matplotlib, multi_image_show_matplotlib
from matplotlib.widgets import Slider

def look_at_dataset_warped(images: str):
    """
    Shows every image's warped perspective that is inside the given directory. Assumes image in image#number.file_extension format.

    Parameters:
        images(str): list of paths to images.
    Returns:
        Nothing
    
    Raises:
        Nothing
    """
    for image in images:
        img = SudokuImage(image)
        print(img.shortened_filename)
        try:
            board, board_binary, board_size = img.find_board_location()
            multi_image_show_matplotlib([board, board_binary], 2, 1)
        except:
            # Doesn't raise an Exception in order to debug bad images faster.
            print(f"{img.shortened_filename}-problematic")
            continue

def configure_threshold(images: str, default_blocksize: int = 53, default_c: int = 7):

    """
    Creates a window with multiple images for manual adaptive threshold configuration. 
    Assumes image in image#number.file_extension format.

    Parameters:
        images(str): list of paths to images.
    Returns:
        Nothing
    Raises:
        Nothing
    """

    for image in images:
        img = SudokuImage(image, default_blocksize, default_c)
        # Need to convert color space as opencv reads images as BGR, but matplotlib wants RGB images. Without changing color space colors are wonky.
        img.img = cv.cvtColor(img.img, cv.COLOR_BGR2RGB)
        img_binarized = img.binarize_image(img.img.copy())
        _, warped_board_binary, _ = img.find_board_location()

        try:
            fig, axes = plt.subplots(ncols=2)
            fig.canvas.manager.full_screen_toggle()
            axes_img1 = axes[0].imshow(img_binarized, cmap='gray', interpolation='none')
            axes_img2 = axes[1].imshow(warped_board_binary, cmap='gray', interpolation='none')
            axes_imgs = [axes_img1, axes_img2]
            plt.subplots_adjust(bottom=0.35)

            cfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
            c_slider = Slider(ax=cfreq, label='c', valmin=0, valmax=40, valinit=default_c, valstep=1,)

            blocksizefreq = fig.add_axes([0.25, 0.3, 0.65, 0.03])
            bs_slider = Slider(ax=blocksizefreq, label='Blocksize', valmin=1, valmax=300, valinit=default_blocksize, valstep=[i for i in range(1, 300, 2)],)
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
            print(f"{img.shortened_filename}-problematic")
            print(e)
            continue
            
def update_c(img, axes_imgs, val):
    img.c = val
    axes_imgs[0].set_data(img.binarize_image(img.img.copy()))
    axes_imgs[1].set_data(img.find_board_location()[1])
def update_bs(img, axes_imgs, val):
    img.blocksize = val
    axes_imgs[0].set_data(img.binarize_image(img.img.copy()))
    axes_imgs[1].set_data(img.find_board_location()[1])



