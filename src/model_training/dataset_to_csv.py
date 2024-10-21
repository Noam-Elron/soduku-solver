from model_training.PairsDatasetManager import ImgDatDirectoryManager
from typing import List, Tuple, Union
from sudoku_logic.sudoku_scanner import SudokuImage
import pandas as pd


def sudoku_attach_labels(img: str, labels: List[str]) -> List[Union[str, int]]:
    """
    Takes an image of a sudoku board and attaches each label to the appropriate flattened(instead of 2D img, flattened to 1D) cell

    Parameters:
        img(str) -- Filepath to an Image of a sudoku board. \n
        labels(List[str]) -- Array of digits in string form, each digit corresponds corresponds to the digit appearing in the corresponding cell position in the cells list. 

    Returns:
        2D Array where each element is an array, the first element in the array is a string that represents the digit appearing and the rest are the flattened pixels of the cell image. 
    """
    # List concatenation essentially appends labels[i] to the start of the cells list.
    img = SudokuImage(img)
    cells = img.return_all_cells()
    return [[labels[i]] + cell.tolist() for i, cell in enumerate(cells)]

def simplify_sudoku_files_to_data(directoryManager: ImgDatDirectoryManager) -> List[Union[str, List[int]]]:
    """ 
    Transforms the images and data files from the dataset to data appropriate to a csv file.

    Parameters:
        None

    Returns:
        Returns a 2D array where each element is a list containing a label and 784 pixels which make up an image of a 28*28 cell
    
    Raises:
        Nothing
    """

    df_data = []
    pairs = directoryManager.get_all_img_dat_files()
    for img, data in pairs:
        sudoku_numbers = directoryManager.get_image_labels(data)
        df_data += sudoku_attach_labels(img, sudoku_numbers)

    return df_data


def create_df(all_data: List[Union[str, int]]):
        """
        Creates pandas dataframe using given data

        Parameters:
            all_data - 2D array of elements that contain a label and 784 pixels.

        Returns:
            Pandas dataframe object
        
        Raises:
            Nothing
        """

        row_names = ["label"] + [f"pixel{i}" for i in range(0, 28*28)]
        df = pd.DataFrame(all_data, columns=[*row_names])
        return df

    
def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combines two dataframes along the y axis

    Parameters:
        df1(pd.Dataframe): Dataframe to be at the beginning of the new Dataframe
        df2(pd.Dataframe): Dataframe to be at the end of the new Dataframe
    Returns:
        pd.Dataframe: New combined Dataframe created from stacking df2 below df1.
    """
    return pd.concat([df1, df2], axis=0)


def output_dataset_csv(directory: str, name: str):
    """Create csv from dataset.

    Args:
        directory (str): Directory to place the output csv file
        name (str): Name for the csv file
    """

    data = simplify_sudoku_files_to_data()
    df = create_df(data)
    df.to_csv(f'./{directory}/{name}.csv', index=False)

