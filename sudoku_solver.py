import time
import copy

# Highlight - CTRL + ALT + H
# Remove Highlight - CTRL + ALT + R

"""
 A1 A2 A3| A4 A5 A6| A7 A8 A9    
 B1 B2 B3| B4 B5 B6| B7 B8 B9    
 C1 C2 C3| C4 C5 C6| C7 C8 C9    
---------+---------+---------    
 D1 D2 D3| D4 D5 D6| D7 D8 D9    
 E1 E2 E3| E4 E5 E6| E7 E8 E9    
 F1 F2 F3| F4 F5 F6| F7 F8 F9    
---------+---------+---------    
 G1 G2 G3| G4 G5 G6| G7 G8 G9    
 H1 H2 H3| H4 H5 H6| H7 H8 H9    
 I1 I2 I3| I4 I5 I6| I7 I8 I9    


 A Sudoku puzzle is a grid of 81 squares. The columns are labeled by number 1-9, the rows are labeled by the letters A-I.  
 A collection of nine squares (can be either a column, row, or box) are called a unit. Squares that share a unit are called peers. 
 Every square has exactly 3 units and 20 peers.
 
 Example of units and peers for A2.

    A2   |         |            A1 A2 A3| A4 A5 A6| A7 A8 A9   A1 A2 A3|         |         
    B2   |         |                    |         |            B1 B2 B3|         |         
    C2   |         |                    |         |            C1 C2 C3|         |         
---------+---------+---------  ---------+---------+---------  ---------+---------+---------
    D2   |         |                    |         |                    |         |         
    E2   |         |                    |         |                    |         |         
    F2   |         |                    |         |                    |         |         
---------+---------+---------  ---------+---------+---------  ---------+---------+---------
    G2   |         |                    |         |                    |         |         
    H2   |         |                    |         |                    |         |         
    I2   |         |                    |         |                    |         |         

Full example:

 A1 A2 A3| A4 A5 A6| A7 A8 A9    
 B1 B2 B3|         |           
 C1 C2 C3|         |     
---------+---------+---------    
    D2   |         |     
    E2   |         |     
    F2   |         |     
---------+---------+---------    
    G2   |         |     
    H2   |         |     
    I2   |         |     

"""

"""
class Board:
    def __init__(self):
"""

def cross(A, B):
    "Cross product of elements in A and elements in B."
    """
    When list comprehensions include two loops, its exactly the same as the scenario below:
    Same as:
    for a in A:
        for b in B:
            list.append(a+b)

    """
    return [a+b for a in A for b in B]

rows     = 'ABCDEFGHI'
digits = '123456789'
cols = digits
# Creates all squares - combinations of A1 up to I9
squares  = cross(rows, cols)
# Each unit inside column_unit is a list that contains every letter added with a single number 
# cross(A, B) receives a single char(digit) at a time into B. Each letter from rows is added with that single digit. It does this for all digits 1-9. 
# The output is a 2D array in which each array inside it represents a column [A1, B1, C1...] till [A9, B9, C9...].
column_unit = [cross(rows, number) for number in cols]

# Each unit inside row_unit is a list that contains a single letter added with every number 
# cross(A, B) receives a single char(letter) at a time into A. That letter is added with each number from cols. It does this for all letters A-I.
# The output is a 2D array in which each array inside it represents a row [A1, A2, A3...] till [I1, I2, I3...].
row_unit = [cross(letter, cols) for letter in rows]

# Each square unit is a combination of Three letters with Three numbers. 
# Output of square unit is a 2D array where each array inside it represents a square unit. [A1, A2, A3, B1, B2, B3...] till [G7, G8, G9, H7, H8, H9...] 
square_unit = [cross(row, col) for row in ("ABC", "DEF", "HGI") for col in ("123", "456", "789")]


unit_list = (column_unit + row_unit + square_unit)
units = dict((s, [u for u in unit_list if s in u]) for s in squares)

# Creates a dictionary mapping each square with a single set that contains all its peers.
"""
 sum(iterable, start) == sum iterates through an iterable and adds each element to start. Default of start is 0.
 sum([[1,2,3], [4, 5, 6]], []]) flattens arrays because sum goes into the iterable and adds each item into the value of start. 
 If start is a list, and the items inside the iterable are lists, then it simply adds lists to lists. 
 Adding lists concatenates them together [1] + [2, 3] = [1, 2, 3]
 """

# For each square creates a tuple that contains two elements: the square and a set that contains all the squares which share a unit with it(peers) without the square itself
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in squares)

def tests():
    "A set of unit tests."
    assert len(squares) == 81
    assert len(unit_list) == 27
    assert all(len(units[s]) == 3 for s in squares)
    assert all(len(peers[s]) == 20 for s in squares)
    assert units['C2'] == [['A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2'],
                           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'],
                           ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']]
    assert peers['C2'] == set(['A2', 'B2', 'D2', 'E2', 'F2', 'G2', 'H2', 'I2',
                               'C1', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                               'A1', 'A3', 'B1', 'B3'])
    print('All tests pass.')


def create_grid(grid):
    grid = grid.replace(".", "0")
    "Convert grid into a dict where key,value pairs look like: {square: char} with '.' for empty squares."
    grid_squares = [square for square in grid if square in cols or square in "0"]
    assert len(grid_squares) == 81
    return dict(zip(squares, grid_squares))

def constraint_propagation(grid):
    "Try to eliminate impossible answers from solution_grid grid based on inputted grid. If impossible value detected return False"
    # To start, every square has the possibility to be any digit.
    solution_grid = dict((square, cols) for square in squares)
    # Loop over all squares inside the received grid. If a square has a real digit inside it(not "0" or "."), assign that number into
    # The solution_grid grid. If it's not possible, return False. 
    for square, digit in grid.items():
        if digit in digits:
            assigned = assign(solution_grid, square, digit)
            if assigned == False: 
                return False ## (Fail if assign returns False a.k.a can't assign digit to square.)
    return solution_grid

# Assign basically just calls eliminate on every digit. If it works then returns new grid.
def assign(solution_grid: list, square: str, digit: str):
    #Return new solution_grid grid, unless a contradiction is detected then return False.

    values_other_than_digit = solution_grid[square].replace(digit, '')
    # Try to remove all possibilities of numbers other than the number the square already has. 
    if all(eliminate(solution_grid, square, value_other_than_digit) for value_other_than_digit in values_other_than_digit):
        return solution_grid
    else:
        return False

# Removes a digit from a square
def eliminate(solution_grid, square, value_other_than_digit):

    if value_other_than_digit not in solution_grid[square]:
        return solution_grid # Already eliminated

    # Remove each digit other than digit from a square's possibilities 
    solution_grid[square] = solution_grid[square].replace(value_other_than_digit,'')
    
    # Base case
    if len(solution_grid[square]) == 0: # Contradiction: removed last value
        return False 
    
    # (1) If a square is reduced to one value, then eliminate that value from the squares peers.
    elif len(solution_grid[square]) == 1:
        digit_to_remove_from_peers = solution_grid[square]
        if not all(eliminate(solution_grid, s2, digit_to_remove_from_peers) for s2 in peers[square]):
            return False
    
    ## (2) If a unit is reduced to only one place for a value, then put that value there.
    for unit in units[square]:
        # Because value_other_than_digit was just eliminated from squares possibilities, then it has to be possible in at least one other square inside
                
        # Goes over each square inside the current unit and adds it to the list if it contains value_other_than_digit
        d_places = [square for square in unit if value_other_than_digit in solution_grid[square]]
        # If value_other_than_digit doesn't appear anywhere in the unit, there has to be a mistake.
        if len(d_places) == 0:
            return False ## Contradiction: no place for this value
        
        # If value_other_than_digit appears in only one square inside the unit, that means it cant appear in any other square, than value_other_than_digit has to be placed inside that square
        elif len(d_places) == 1:
            # Tries to place value_other_than_digit inside the specific square. 
            if assign(solution_grid, d_places[0], value_other_than_digit) == False:
                return False
    return solution_grid

def display(solution_grid):
    "Display these solution_grid as a 2-D grid."
    width = 1+max(len(solution_grid[square]) for square in squares)
    # Creates a list containing three strings: each string is '-' times 3*width
    line_chars = ['-'*(width*3)]*3
    # Takes each of those three lines and joins them with a '+' in-between each line
    line = ('+'.join(line_chars))
    for row in rows:
        # Takes each square and puts the solution_grid inside it to fit perfectly center inside width, that way everything will be on the same line.
        # If a square contains 3 or 6 then a '|' is printed after it, else nothing.
        print(''.join(solution_grid[row+column].center(width)+('|' if column in '36' else '') for column in cols))
        # If the row is row C or F, then the first or second layer of squares ended and the separation line is printed
        if row in 'CF': 
            print(line)
    """
    #Alternative way
    for square in squares:
        print(solution_grid[square].center(width) + ('|' if '3' in square or '6' in square else ''), end="")
        if ("9" in square):
            print()

        if (square == "C9" or square == "F9"):
            print(line)
    """

def backtracking(solution_grid):
    "Using depth-first search and propagation, try all possible solution_grid."
    if solution_grid is False:
        return False # Failed earlier
    
    # If every single square inside the grid has only one value, grid is solved.
    if all(len(solution_grid[s]) == 1 for s in squares): 
        return solution_grid # Solved!

    # Chose the unfilled square s with the fewest possibilities
    # Go over all the values of the squares, if a square has more than 1 value -> create a tuple of the square with the amount of digits is still has possible. return square with the 
    # least amount of possibilities left
    _ , s = min((len(solution_grid[s]), s) for s in squares if len(solution_grid[s]) > 1)

    # For every digit left in the square from above, try to assign that digit to square, once a digit works assign returns the new grid and backtracking is called again on the
    # New grid recursively
    return some(backtracking(assign(solution_grid.copy(), s, d)) for d in solution_grid[s])

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: 
            return e
    return False

def solve(grid): 
    # Solves the given grid by first creating the constraint_propagated grid, and passing that to be backtracked upon
    return backtracking(constraint_propagation(grid))

""" Second Approach, more Brute-force with just backtracking """
def solve_two(grid):
    # First create a grid that assigns each square to a value. Easier to traverse.
    dict_grid = create_grid(grid) 
    backtracking_solo(dict_grid)
    return dict_grid

def backtracking_solo(grid):
    empty_square = findEmpty(grid) # Returns a string of an empty square
    if (empty_square is None):
        return True
    
    for digit in digits:
        # If a valid digit exists, place it in that square and recursively try the next one.
        if (check_validity(grid, empty_square, digit)):
            grid[empty_square] = digit
            if(backtracking_solo(grid)):
                return True
            grid[empty_square] = '.'
    return False

def findEmpty(current_grid):
    for square in squares:
        if current_grid[square] == '.' or current_grid[square] == '0':
            return square

def check_validity(current_grid, square, digit):
    for peer in peers[square]:
        if digit in current_grid[peer]: 
            return False
    return True

def compare_solutions(solver1_function_grid, solver2_function_grid):
    board1 = timer(solver1_function_grid)
    board2 = timer(solver2_function_grid)

    for square in squares:
        if board1[square] != board2[square]:
            print("Diffrent Solutions!")
            return False
    print("Both Solutions are identical")
    return True

def timer(func):
    start = time.time()
    board = func(copy.deepcopy(grid))
    print(f"Time to solve: {time.time() - start}")
    return board

#grid = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
#grid = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
#compare_solutions(solver1_function_grid= solve, solver2_function_grid= solve_two)

#print(s)
#display(create_grid(grid))
#print()
#display(solve(grid))