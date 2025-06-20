import pygame
import numpy as np
from vis_constants import colors, font, player_colors
import math


class TrainCardMarket:
    def __init__(self, screen_dim, board_height):
        self.card_dim = (35, 50)
        self.card_loc = (screen_dim[0] / 2 - 3 * self.card_dim[0], (screen_dim[1] - board_height) / 2 - self.card_dim[1])
        
        self.train_cards = [pygame.Surface(self.card_dim) for _ in range(6)]
        [train_card.fill("white") for train_card in self.train_cards[:5]] # Initialize as white (invisible)
        self.train_cards[-1].fill("gray") # Closed deck


    def update(self, train_cards):
        assert len(train_cards) == 5, "There should be five open train cards"

        # Update the colors
        [self.train_cards[i].fill(colors[train_cards[i]]) for i in range(5)]


    def draw(self, screen):
        [screen.blit(train_card, (self.card_loc[0] + i * self.card_dim[0], self.card_loc[1])) for i, train_card in enumerate(self.train_cards)]


class PlayerHand:
    def __init__(self, screen_dim, board_dim, player_i):
        self.card_dim = (50, 35)
        self.player_i = player_i

        if self.player_i == 0:
            # Top left of board
            self.card_loc = ((screen_dim[0] - board_dim[0]) / 2 - self.card_dim[0], (screen_dim[1] - board_dim[1]) / 2)
        elif self.player_i == 1:
            # Top right of board
            self.card_loc = ((screen_dim[0] - board_dim[0]) / 2 + board_dim[0], (screen_dim[1] - board_dim[1]) / 2)
        elif self.player_i == 2:
            # Bottom left of board
            self.card_loc = ((screen_dim[0] - board_dim[0]) / 2 - self.card_dim[0], (screen_dim[1] - board_dim[1]) / 2 + board_dim[1] - 9 * self.card_dim[1])
        elif self.player_i == 3:
            # Bottom right of board
            self.card_loc = ((screen_dim[0] - board_dim[0]) / 2 + board_dim[0], (screen_dim[1] - board_dim[1]) / 2 + board_dim[1] - 9 * self.card_dim[1])
        elif self.player_i == 4:
            # Bottom of board
            self.card_dim = (35, 50)
            self.card_loc = (screen_dim[0] / 2 - 4.5 * self.card_dim[0], (screen_dim[1] - board_dim[1]) / 2 + board_dim[1])
        else: 
            raise Exception("Too many players!")
            
        # Colored surfaces
        self.hand = [pygame.Surface(self.card_dim) for _ in range(9)]
        [card.fill(colors[i]) for i, card in enumerate(self.hand)]

        # Card counts
        self.hand_counts = np.zeros((9,))


    def update(self, hand_counts):
        assert isinstance(hand_counts, np.ndarray) and len(hand_counts) == 9, "Inappropriate counts input"
        self.hand_counts = hand_counts

    
    def draw(self, screen):
        # Draw colored cards
        [screen.blit(self.hand[i], (self.card_loc[0], self.card_loc[1] + i * self.card_dim[1])) for i in range(9)]

        for i, count in enumerate(self.hand_counts):
            if i <= 1:
                # Display white text on black or blue card
                text = font.render(str(count), False, "white")
            else:
                # Display black text otherwise
                text = font.render(str(count), False, "black")

            # Draw counts
            if self.player_i in range(4):
                screen.blit(text, (self.card_loc[0] + self.card_dim[0] * 0.5, self.card_loc[1] + self.card_dim[1] * (i + 0.5)))
            elif self.player_i == 4:
                screen.blit(text, (self.card_loc[0] + self.card_dim[0] * (i + 0.5), self.card_loc[1] + self.card_dim[1] * 0.5))


class ActionTracker:
    def __init__(self, loc):
        self.loc = loc
        self.phase_text = font.render("", False, "black")
        self.action_text = font.render("", False, "black")
        self.current_player_text = font.render("", False, "black")
        self.cover_width = 0


    def update(self, phase, action, current_player):
        if phase == 0:
            phase_str = "Initial ticket selection"
        elif phase == 1:
            phase_str = "Main"
        elif phase == 2:
            phase_str = "Draw train cards"
        elif phase == 3:
            phase_str = "Choose tickets"
        elif phase == 4:
            phase_str = "Claim route"
        elif phase == 5:
            phase_str = "Choose payment"

        self.cover_width = max(self.phase_text.get_width(), self.action_text.get_width())
        self.phase_text = font.render(f"Phase: {phase_str}", False, "black")
        self.action_text = font.render(f"Action: {action}", False, "black")
        self.current_player_text = font.render(f"Current player: {current_player}", False, player_colors[current_player])


    def draw(self, screen):
        # First draw white rectangles to cover previous text
        white_surface = pygame.Surface((self.cover_width, font.get_height() * 3))
        white_surface.fill("white")        
        screen.blit(white_surface, self.loc)

        # Then draw text
        screen.blit(self.phase_text, self.loc)
        screen.blit(self.action_text, (self.loc[0], self.loc[1] + font.get_height()))
        screen.blit(self.current_player_text, (self.loc[0], self.loc[1] + 2 * font.get_height()))


class Point:
    def __init__(self, point_t = (0,0)):
        # Constructed using a normal tupple
        self.x = float(point_t[0])
        self.y = float(point_t[1])


    # Define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))
    
    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))
    
    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))
    
    def __truediv__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))
    
    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))
    
    # Get back values in original tuple format
    def get(self):
        return (self.x, self.y)


def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length

    for index in range(0, math.ceil(length/dash_length), 2):
        start = origin + (slope *    index    * dash_length)
        end   = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)