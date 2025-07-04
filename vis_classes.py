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


class ScoreBoard:
    def __init__(self, n_players, loc, traj_len):
        self.n_players = n_players
        self.loc = loc
        self.player_scores = np.zeros((self.n_players, traj_len))
        self.cover_width = 0

    
    def update(self, current_player, step_index, turn_reward):
        if self.player_scores[current_player, step_index] == 0:
            # If not zero, score has already been updated for this step
            self.player_scores[current_player, step_index] = self.player_scores[current_player, step_index - 1] + turn_reward

        for other_player in range(self.n_players):
            if other_player != current_player:
                # Don't update current player
                self.player_scores[other_player, step_index] = self.player_scores[other_player, step_index - 1]

    def draw(self, screen, step_index):
        # First draw white surface to cover previous text
        white_surface = pygame.Surface((self.cover_width, font.get_height() * self.n_players))
        white_surface.fill("white")
        screen.blit(white_surface, self.loc)

        # Then draw text
        player_texts = [font.render(f"Player {i} score: {score[step_index]}", False, "black") for i, score in enumerate(self.player_scores)]
        [screen.blit(player_text, (self.loc[0], self.loc[1] + i * font.get_height())) for i, player_text in enumerate(player_texts)]

        self.cover_width = max([player_text.get_width() for player_text in player_texts])


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
