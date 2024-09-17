import pygame 
from src.sudoku_logic.sudoku_solver import backtracking, constraint_propagation, create_grid, check_validity, findEmpty  
from sudoku_scanner import FileDialogWindow, SudokuImage
import time 

def gui(board):
    # GLOBALS
    SCREEN_WIDTH = 700
    SCREEN_HEIGHT = 625
    colors: dict = {"white": (255, 255, 255), "black": (0, 0, 0), "orange": (255, 189, 3), "red": (255, 0, 0), 
    "l-blue": (173, 216, 230)}

    # SUDOKU STUFF FOR CONVIENCE OF ITERATING AND STUFF
    def cross(A, B):
        return [a+b for a in A for b in B]

    rows     = 'ABCDEFGHI'
    digits = '123456789'
    cols = digits
    squares  = cross(rows, cols)


    # Initialize Pygame stuff
    pygame.init()
    fps = 60
    frame_counter = pygame.time.Clock()
    screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    screen.convert()
    # Makes the screen go white maybe slightly faster when done here too?
    screen.fill(colors["white"])
    pygame.display.set_caption('Sudoku')
    timer_font: pygame.font.SysFont = pygame.font.SysFont("garamond", 28, bold = True)
    solve_button_font: pygame.font.SysFont = pygame.font.SysFont("garamond", 24, bold=True)


    # CLASSES

    class Grid:
        rows = 9
        cols = 9
        def __init__(self, board, width: int, height: int):
            self.board = create_grid(board)
            # Added 11 for gap space.
            self.width = width + 11
            self.height = height + 11
            self.boardX = ((SCREEN_WIDTH-self.width)//2)
            self.boardY = 0.25*((SCREEN_HEIGHT-self.height)//2) 
            self.grid_rect = None
            self.squares = [Square((self.width-11) // 9, (self.height-11) // 9, square, self.board[square], self.boardX, self.boardY) for square in self.board]

            self.squares_map = dict((self.squares[square].pos, self.squares[square]) for square in range(len(self.squares)))
            self.square_positions_map = dict((self.squares[square], (self.squares[square].x, self.squares[square].y)) for square in range(len(self.squares)))
            
            self.buttons_dict = {}

        def update_squares(self):
            # Square takes in position(square -- "A1", "C5" etc) and value inside.
            self.squares = [Square((self.width-11) // 9, (self.height-11) // 9, square, self.board[square], self.boardX, self.boardY) for square in self.board]
            self.squares_map = dict((self.squares[square].pos, self.squares[square]) for square in range(len(self.squares)))

        def grid_draw(self):
            self.grid_rect = pygame.draw.rect(screen, colors["black"], ( self.boardX, self.boardY, self.width, self.height), width = 5)
            for i in range(1, 3):
                # Vertical Lines
                pygame.draw.line(screen, colors["black"], (self.boardX + (self.width//3) * i, self.boardY ), (self.boardX+(self.width//3) * i, self.boardY + self.height - 1), width = 5)
                # Horizontal Lines
                pygame.draw.line(screen, colors["black"], (self.boardX, self.boardY + (self.height//3) * i), (SCREEN_WIDTH - self.boardX - 2, self.boardY + (self.height//3) * i), width = 5)
            
        def draw_squares(self):
            for i in range(len(self.squares)):
                self.squares[i].draw_square()

        # SOLVE BUTTONS
        def buttons(self):
            solve_button_slow = pygame.draw.rect(screen, colors["orange"], (self.boardX, 1.025*(self.boardY+self.height), 0.25*self.width, 0.1*self.height))
            solve_button_fast = pygame.draw.rect(screen, colors["orange"], (self.boardX + 2 * (self.width//3) - 0.25*self.width, solve_button_slow.y, solve_button_slow.width, solve_button_slow.height))
            clear_button = pygame.draw.rect(screen, colors["l-blue"], ((solve_button_slow.right + solve_button_fast.left - (solve_button_slow.width//1.75))//2 + 1 , solve_button_slow.y, solve_button_slow.width//1.75 ,solve_button_slow.height))

            self.buttons_dict = {"slow": solve_button_slow, "fast": solve_button_fast, "clear": clear_button}
            for i in range(2):
                solve_text = solve_button_font.render(f"Solve - {list(self.buttons_dict)[i].capitalize()}", True, colors["black"])
                solve_text.convert()
                solve_rect = solve_text.get_rect()
                solve_rect.center = list(self.buttons_dict.values())[i].center
                screen.blit(solve_text, solve_rect)
            
            clear_button_text = solve_button_font.render("Clear", True, colors["black"])
            clear_button_text.convert()
            clear_button_rect = clear_button_text.get_rect()
            clear_button_rect.center = clear_button.center
            screen.blit(clear_button_text, clear_button_rect)
        
        def solve_slow(self, time_start):
            empty_square = findEmpty(self.board) # Returns a string of an empty square
            if (empty_square is None):
                return True
            pygame.event.pump()
            for digit in digits:
                # If a valid digit exists, place it in that square and recursively try the next one.
                if (check_validity(self.board, empty_square, digit)):
                    # Change digit inside board, board is passed to next recursive call
                    self.board[empty_square] = digit
                    # Change actual square digit, the digit inside Square object is what is used to draw the square.
                    self.squares_map[empty_square].digit = digit
                    update_screen(self, time_start)
                    pygame.display.flip()
                    if(self.solve_slow(time_start)):
                        return True
                    self.board[empty_square] = '0'
            return False

        def solve_fast(self):
            self.board = backtracking(constraint_propagation(self.board))
            self.update_squares()

        def clear(self, original_board):
            self.board = create_grid(original_board)
            self.update_squares()

    class Square:
        def __init__(self, width: int, height: int, pos: str, digit: str, boardX: int, boardY: int):
            self.width = width
            self.height = height
            # Square name, ex -- "A5"
            self.pos = pos
            # Digit inside Square
            self.digit = digit
            
            self.boardX = boardX
            self.boardY = boardY
            self.index = squares.index(self.pos)
            # Since Squares list contains all the squares in order of row, dividing the index by 9 will yield a result of row and a leftover of column. For example square "A5" which is at index 4 so - 4//9, 4 divided by nines yields a result of 0 and a column of 4. So first row, fifth column.
            self.row, self.col = divmod(self.index, 9)
            self.notes = None
            self.coords = self.pos_square_correctly()
            # Gotta add 4 to x and y cuz of width of outside big box
            self.x = self.coords[0] + 4
            self.y = self.coords[1] + 4

        def draw_square(self):
            square_rect = pygame.draw.rect(screen, colors["black"], (self.x, self.y, self.width, self.height), width = 1)
            if self.digit != "0":
                square_num_text = solve_button_font.render(f"{self.digit}", True, colors["black"])
                square_num_text.convert()
                text_rec = square_num_text.get_rect()
                text_rec.center = square_rect.center
                screen.blit(square_num_text, text_rec)

        def pos_square_correctly(self):
            self.x, self.y = self.boardX + self.width * self.col, self.boardY + self.height * self.row
            if (self.pos[0] in ["D", "E", "F"]):
                self.y += 4
            elif (self.pos[0] in ["G", "H", "I"]):
                self.y += 8
            if (self.pos[1] in ["4", "5", "6"]):
                self.x += 4
            elif (self.pos[1] in ["7", "8", "9"]):
                self.x += 8
            return (self.x, self.y)


    # FUNCTIONS
    def timer(grid, starting_time):
        timer_text = timer_font.render(f"Time: {format_time(round(time.time() - starting_time))}", True, colors["black"])
        timer_text.convert()
        timer_rect = timer_text.get_rect()
        timer_rect.center = (0.75*SCREEN_WIDTH, 1.07*(grid.boardY+grid.height))
        screen.blit(timer_text, timer_rect)

    def format_time(secs):
        sec = secs%60
        minute = secs//60

        formatted = str(minute) + "m" + ", " + str(sec) + "s"
        return formatted




    def start(grid):
        grid.grid_draw()
        grid.draw_squares()
        grid.buttons()
    def update_screen(grid, starting_time):
            screen.fill(colors["white"])
            start(grid)
            timer(grid, starting_time)

    def main(board):
        #board = "000003017015009008060000000100007000009000200000500004000000020500600340340200000"
        #board = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
        #board = "8..........36......7..9.2...5...7.......457.....1...3...1....68..85...1..9....4.."
        grid = Grid(board, 500, 500)
        time_start = time.time()

        while True: 
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if grid.buttons_dict["slow"].collidepoint(pos):
                        print("Slow")
                        grid.solve_slow(time_start)
                    elif(grid.buttons_dict["fast"].collidepoint(pos)):
                        print("Fast")
                        grid.solve_fast() 
                    elif(grid.buttons_dict["clear"].collidepoint(pos)):
                        print("Clear")
                        grid.clear(board)

            update_screen(grid, time_start)
            frame_counter.tick()
            pygame.display.flip()
        pygame.quit()

    main(board)

if __name__ == "__main__":
    gui("000003017015009008060000000100007000009000200000500004000000020500600340340200000")
    