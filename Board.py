import pygame 

"""
Surfaces:
1. Screen
2. Background
3. ?
"""
# GLOBALS

colors: dict = {"white": (255, 255, 255), "black": (0, 0, 0)}

def pygame_setup(screenX: int, screenY: int) -> pygame.Surface:
    # Initialize Pygame to begin
    pygame.init()
    # Create a pygame.Surface Object with inputted screen size
    screen: pygame.Surface = pygame.display.set_mode((screenX, screenY))
    # Draw stuff on background then return background object
    background_draw(screen)
    print(screen)
    pygame.display.set_caption('Sudoku')
    return screen

def background_draw(screen: pygame.Surface):
    # Append background to screen at position 0, 0.
    pygame.draw.rect(screen, (255, 0, 0), (10, 50, 100, 100))

screen = pygame_setup(900, 700)

def main():
    while True: 
        
        for event in pygame.event.get():
            #print(event)
            if event.type == pygame.QUIT:
                return
        screen.fill(colors["white"])
        pygame.display.update()
        
    pygame.quit()



if __name__ == "__main__":
    main()