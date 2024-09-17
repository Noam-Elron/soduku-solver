import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import os
import re

# MANUAL IMAGE SEGMENTATION ADAPTIVE THRESHOLDING PARAMATER FINE TUNING TOOL


cmd = "wmic path Win32_VideoController get CurrentVerticalResolution,CurrentHorizontalResolution"
size_tuple = tuple(map(int,os.popen(cmd).read().split()[-2::]))
screen_width = size_tuple[0]
screen_height = size_tuple[1]
#print(size_tuple)

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

            self.filenames = fd.askopenfilenames(
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



class Slider:
    def __init__(self, parent, images):
        self.parent = parent
        self.images = images

        self.blocksize = 3
        self.c = 1
        parent.title("Slider")
        parent.geometry('600x150')
        # Place window at 100, 100
        parent.geometry(f'+{(screen_width)//2-600}+0')
        parent.resizable(False, False)

        self.scale_blocksize = tk.Scale(parent, from_=3, to=40 ,tickinterval=2, length = 600, variable=self.blocksize, orient='horizontal', command= lambda val: self.return_vals("b", val))
        self.scale_c = tk.Scale(parent, from_=1, to=30, tickinterval=5, length = 600, variable=self.c, orient='horizontal', command= lambda val: self.return_vals("c", val))
        self.scale_blocksize.pack()
        self.scale_c.pack()

    def return_vals(self, func_name, val):
        val = int(val)
        if func_name == "b" and (val % 2 == 1):
            self.blocksize = val
        elif func_name == "c":
            self.c = val
        for i in range(len(self.images)):
            self.images[i].image_show(self.blocksize, self.c)
        

class Image:
    def __init__(self, filename, width, height, pos_x, pos_y):
        self.filename = filename
        self.width = width
        self.height = height
        self.pos_x = pos_x
        self.pos_y = pos_y
        
        self.img = cv.imread(self.filename)

    def preprocess(self, blocksize, c):
        self.img = cv.imread(self.filename)
        # Convert image pixels to Grayscale
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        # Resize image, unsure if want to keep scale
        self.img = cv.resize(self.img, (self.width, self.height))
        self.img = cv.adaptiveThreshold(self.img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blocksize, c)
        self.img = cv.bitwise_not(self.img)

    def image_show(self, blocksize, c):
        self.preprocess(blocksize, c)
        cv.imshow(f"{self.filename}", self.img)      
        cv.moveWindow(f"{self.filename}", self.pos_x, self.pos_y)
        
     
# My screen size is 1280x720 according to opencv?
def main():
    win = MainWindow()
    win.file_dialog()
    #print(win.filenames[0])
    image_width, image_height = 400, 400
    # Limit images to 3 to prevent overflow
    image_start = 40
    if len(win.filenames) == 1:
        image_width, image_height = 600, 600
        image_start = 1920//2-image_width
    if len(win.filenames) == 2:
        image_width, image_height = 500, 500
        image_start = 1920//3-image_width
    images = [Image(file, image_width, image_height, image_start+image_width*i, 180) for i, file in enumerate(win.filenames[:3])]
    
    

    slider_window = tk.Tk()
    slider = Slider(slider_window, images)
    slider_window.mainloop()


    
if __name__ == "__main__":
    main()

    