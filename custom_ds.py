from fileprompter import FileDialogWindow
import cv2 as cv
def main():
    win = FileDialogWindow()
    img = cv.imread(win.filename)
    print(type(img))

if __name__ == "__main__":
    main()