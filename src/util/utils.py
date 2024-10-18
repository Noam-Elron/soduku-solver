import cv2 as cv

def pad_image(image, pixels: int, color: int):
    # args are src, top padding, bottom padding, left padding, right padding, border type, border value/color if type is constant.
    image = cv.copyMakeBorder(image, pixels, pixels, pixels, pixels, cv.BORDER_CONSTANT, color)
    return image


