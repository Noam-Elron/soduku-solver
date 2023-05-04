import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
import os 

class FileDialogWindow:
    """
    Opens a file dialog window

    Parameters:
        Optional: 
            Directory -- Either simply directory name if directory in cwd or full path if root is set to True.
            Root -- Boolean, True if directory exists outside cwd.

    Returns:
        Returns a FileDialogWindow instance, instance contains filename instance variable.
    
    Raises:
        Nothing
    """

    # TODO - Create failsafes incase direcory entered incorrectly or root used incorrectly.
    root = tk.Tk()
    root.title('Tkinter Open File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    def __init__(self, directory=None, root=False):
        self.filename = None
        self.directory = directory
        self.root = root
        self.path = self.get_path()
        self.file_dialog()

    def get_path(self):
        if self.directory != None and self.root != False:
            path = self.directory
        elif self.directory != None:
            path = os.path.join(os.getcwd(), self.directory)
        else:
            path = os.getcwd()
        print(path)
        return path

    def file_dialog(self):
        

        def select_files():
            filetypes = (
                ('text files', '*.jpg *.png *.jpeg'),
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

    def __repr__(self):
        return f"{__class__}({self.directory}, {self.root})"