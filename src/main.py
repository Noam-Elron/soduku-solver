from sudoku_logic.sudoku_scanner import FileDialogWindow, SudokuImage
from sudoku_logic.sudoku_solver_gui import gui

def main():
    win = FileDialogWindow(directory="images")
    blocksize, c = 23, 7
    image = SudokuImage(win.filename)    
    image.return_board()
    grid = image.return_board()
    gui(grid)
    

if __name__ == "__main__":
    main()