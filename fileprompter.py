import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import os 

class FileDialogWindow:
    root = tk.Tk()
    root.title('Tkinter Open File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    def __init__(self, directory=None):
        self.filename = None
        self.directory = directory
        self.path = self.get_path()
        self.file_dialog()

    def get_path(self):
        if self.directory != None:
            path = os.path.join(os.getcwd(), self.directory)
        else:
            path = os.getcwd()
        print(path)
        return path

    def file_dialog(self):
        

        def select_files():
            filetypes = (
                ('text files', '*.jpg *.png'),
                ('All files', '*.*')
            )
            # Normalize os path so that it auto configures \ for windows and // for linux
            self.filename = os.path.normcase(fd.askopenfilename(
                title='Open a file',
                initialdir=self.path,
                filetypes=filetypes))

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
