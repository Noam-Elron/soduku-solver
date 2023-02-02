import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

class FileDialogWindow:
    root = tk.Tk()
    root.title('Tkinter Open File Dialog')
    root.resizable(False, False)
    root.geometry('300x150')
    def __init__(self):
        self.filename = None
        self.file_dialog()

    def file_dialog(self):
        

        def select_files():
            filetypes = (
                ('text files', '*.jpg *.png'),
                ('All files', '*.*')
            )

            self.filename = fd.askopenfilename(
                title='Open a file',
                initialdir='D:\Downloads\Creation\PythonCreations\soduku-solver\images',
                filetypes=filetypes)

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
