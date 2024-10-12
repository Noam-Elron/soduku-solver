import math

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import cv2 as cv


def image_show_matplotlib(image, title="Image"):
    fig = plt.figure()
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title(title)
    # Removes x ticks from plot(because passing empty list)
    plt.xticks([])
    # Removes y ticks from plot(because passing empty list)
    plt.yticks([])
    plt.show()

def multi_image_show_matplotlib(images, num_images, num_in_row):
    """
    Show multiple images in same matplotlib figure. 
    """
    rows = math.ceil(num_images/num_in_row)
    fig, axis = plt.subplots(nrows=rows, ncols=num_in_row, figsize=(12, 14))
    axis = axis.flatten()
    axis = axis[:num_images]
    for i, ax in enumerate(axis.flat):
        ax.imshow(images[i], cmap='gray', interpolation='none')
        ax.set(title = f"Image #{i+1}")
    plt.show()