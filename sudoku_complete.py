from sudoku_scanner import FileDialogWindow, SudokuImage
from sudoku_solver_gui import gui

def main():
    win = FileDialogWindow("images")
    blocksize, c = 23, 7
    image = SudokuImage(win.filename)    
    grid = image.return_board()
    gui(grid)

if __name__ == "__main__":
    main()