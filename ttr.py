# %%

from typing import Optional
import gymnasium as gym
import numpy as np
import pandas as pd
import ast
import networkx as nx
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygame

# Constants for action definitions
ACTION_DRAW_TRAIN_CARDS = 0
ACTION_DRAW_TICKETS = 1
ACTION_CLAIM_ROUTE = 2

# Constants for phases
PHASE_INITIAL_TICKET_SELECTION = 0
PHASE_MAIN = 1
PHASE_DRAW_TRAIN_CARDS = 2
PHASE_CHOOSE_TICKETS = 3
PHASE_CLAIM_ROUTE = 4
PHASE_CHOOSE_PAYMENT = 5

# Routes array indices
COST_ID_INDEX = 2
CLAIMED_BY_INDEX = 3
DRAW_BONUS_INDEX = 5
LENGTH_INDEX = 6
COLOR_INDEX = 7
GAY_CARDS_INDEX = 8
PAYMENT_IDS_INDEX = 9

# Player tickets indices
TICKET_OWNED = 0
TICKET_COMPLETED = 1

# Trajectory indices
STATE_TENSOR_INDEX = 0
REWARD_INDEX = 1
VALUE_INDEX = 2
LOG_ACTION_PROBS_INDEX = 3
ACTION_INDEX = 4
PHASE_INDEX = 5
ACTION_MASK_INDEX = 6

# %%

class TicketToRideNorthernLightsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_players: int = 2):
        self.n_players = n_players # Number of players in the game
        self._current_player = 0

        self._n_tickets = 55 # Number of tickets to draw from
        self._n_starting_trains = 40 # Number of trains each player starts with
        self._n_train_col = 9 # Number of different train card types 
        self._train_card_deck = TrainCardDeck(self.np_random) # Deck of train cards

        # Scoring system for route length
        self._length_to_points = {
            1: 1,
            2: 2,
            3: 4,
            4: 7,
            5: 10,
        }

        # Possible actions for keeping tickets (must keep at least one, or two for the initial ticket selection)
        self._choose_ticket_options = [opt for opt in list(itertools.product([0, 1], repeat=3)) if sum(opt) >= 1]
        self._choose_initial_ticket_options = [opt for opt in list(itertools.product([0, 1], repeat=4)) if sum(opt) >= 2]

        # Define game definitions from excel file
        game_definitions = pd.read_excel("game_definitions.xlsx", sheet_name=None)
        self._card_id_to_color = game_definitions["card_id_to_color"].set_index("id")["color"].to_dict()
        self._city_id_to_name = game_definitions["city_id_to_name"].set_index("id")["name"].to_dict()
        self._country_id_to_name = game_definitions["country_id_to_name"].set_index("id")["name"].to_dict()
        self._country_id_to_city_ids = game_definitions["country_id_to_city_ids"].set_index("country_id")["city_ids"].apply(ast.literal_eval).to_dict()
        self._tickets = game_definitions["tickets"].to_numpy()[:, 1:] # Ticket information, drop id column
        self._route_cost = game_definitions["route_cost"]
        self._route_cost["payment_ids"] = self._route_cost["payment_ids"].apply(ast.literal_eval)
        self._route_cost_payments = game_definitions["route_cost_payments"].to_numpy()[:, 1:] # Payment options, drop id column
        self._routes = game_definitions["routes"] # This is part of the state with a "claimed" value
        self._routes = pd.merge(self._routes.drop(columns="id"), self._route_cost, left_on="cost_id", right_on="id", how="left").drop(columns=["id"]).to_numpy()[:, :-1].astype(int) # Drop payment ids column
        
        # Graph of the board for easy ticket completion check
        self._board_graph = nx.Graph() 
        self._board_graph.add_nodes_from(self._city_id_to_name.keys())
        
        assert len(self._card_id_to_color) == self._n_train_col, "Mismatch between color list and n_train_col"
        assert (len(self._country_id_to_city_ids.keys()) == len(self._country_id_to_name)) & (sum([len(v) for v in self._country_id_to_city_ids.values()]) == len(self._city_id_to_name)), "Mismatch in city/country lists"

        # Define other state variables
        self._train_cards = np.zeros((5,), dtype=int) # The five open train cards
        self._player_hands = np.zeros((self.n_players, self._n_train_col), dtype=int) # Card counts that players are holding
        self._player_tickets = np.zeros((self.n_players, self._n_tickets, 2), dtype=int) # Binary vectors per player to signal which tickets they're holding
        self._remaining_trains = np.full((self.n_players,), self._n_starting_trains, dtype=int) # Vector of values which the number of trains players have left
        self._initial_ticket_selection_done = np.zeros(self.n_players, dtype=bool)
        self._final_round = False
        self._turns_remaining = 0
        self._route_to_claim = np.array([], dtype=int)


        # Define action space(s)
        self._phase = 0 # Will change based on the action the agent chose
        self._pending_tickets = np.full((self.n_players, 4), -1) # Temporary storage for drawn tickets

        self._main_action_space = gym.spaces.Discrete(3) # First choose one of draw train cards, draw tickets, claim route
        self._draw_train_action_space = gym.spaces.Discrete(6) # Pick one of 5 face-up, or the face-down card
        self._choose_ticket_action_space = gym.spaces.Discrete(len(self._choose_ticket_options)) # Choose a number of tickets to keep
        self._choose_initial_tickets_action_space = gym.spaces.Discrete(len(self._choose_initial_ticket_options)) # Player chooses from four tickets initially
        self._claim_route_action_space = gym.spaces.Discrete(len(self._routes)) # Choose one of the routes to claim
        self._choose_payment_action_space = gym.spaces.Discrete(len(self._route_cost_payments)) # Choose a way to pay

        self.action_space = self._choose_initial_tickets_action_space # Initialize action space
        self._biggest_action_spaces_dim = max([gym.spaces.flatten_space(space).shape[0] for space in [self._main_action_space, self._draw_train_action_space, self._choose_initial_tickets_action_space, self._choose_ticket_action_space, self._claim_route_action_space]])

        # Define observation space
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": gym.spaces.MultiBinary(self._biggest_action_spaces_dim),
                "current_player": gym.spaces.Discrete(self.n_players), # For shared policy training, agent must know its identity
                "pending_tickets": gym.spaces.Box(low=0, high=self._n_tickets, shape=(4,), dtype=int), # Tickets to choose from upon drawing tickets
                "phase": gym.spaces.Discrete(4), # Main, draw_train_card, choose_tickets, or claim_route
                "player_hand": gym.spaces.Box(low=0, high=18, shape=(self._n_train_col,), dtype=int), # Encode the card count per card type
                "player_tickets": gym.spaces.Box(low=np.repeat([[0, 0, 0, 0, 2]], self._n_tickets, axis=0), high=np.repeat([[1, 1, 49, 49, 19]], self._n_tickets, axis=0), shape=(self._n_tickets, self._tickets.shape[1] + 2), dtype=int),
                "remaining_trains": gym.spaces.Box(low=0, high=self._n_starting_trains, shape=(self.n_players,), dtype=int),
                "routes": gym.spaces.Box(low=np.repeat([[0, 0, 1, -1, 0, -1, 0, 0]], self._routes.shape[0], axis=0), high=np.repeat([[49, 49, 5, 9, 2, self.n_players, 1, 3]], self._routes.shape[0], axis=0), shape=(len(self._routes), 8), dtype=int), # Overview of all routes
                "train_cards": gym.spaces.Box(low=0, high=self._n_train_col - 1, shape=(5,), dtype=int), # The five open cards, can each be 0-8 to signal which card                
            }
        )

        # Rendering variables
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
    def step(self, action):
        if self._phase == PHASE_INITIAL_TICKET_SELECTION:
            # Must first select at least two tickets from intial four
            chosen_tickets, reward = self._choose_tickets(action, initial=True)
            self._player_tickets[self._current_player, chosen_tickets] = 1
            self._initial_ticket_selection_done[self._current_player] = 1

            if np.all(self._initial_ticket_selection_done == 1):
                # Everyone completed first action, move on to main phase
                terminate = self._end_turn()

                return self._get_obs(), 0, terminate, False, {}
            else:
                # Stay in this phase
                self._current_player = (self._current_player + 1) % self.n_players
            
                return self._get_obs(), 0, False, False, {} 

        elif self._phase == PHASE_MAIN:
            if action == ACTION_DRAW_TRAIN_CARDS:
                # Intermediate "action" to go into the proper phase
                self._phase = PHASE_DRAW_TRAIN_CARDS
                self._n_cards_drawn = 0
                self.action_space = self._draw_train_action_space

                return self._get_obs(), 0, False, False, {} # Observation, reward, terminated, truncated, info
            
            elif action == ACTION_DRAW_TICKETS:
                # Draw tickets and change the phase to make the next action be about choosing tickets
                self._phase = PHASE_CHOOSE_TICKETS
                self._pending_tickets[self._current_player, :3] = self._draw_tickets()
                self.action_space = self._choose_ticket_action_space

                return self._get_obs(), 0, False, False, {} # Observation, reward, terminated, truncated, info
            
            elif action == ACTION_CLAIM_ROUTE:
                # Intermediate "action" to go into the proper phase
                self._phase = PHASE_CLAIM_ROUTE
                self.action_space = self._claim_route_action_space

                return self._get_obs(), 0, False, False, {} # Observation, reward, terminated, truncated, info
            
        elif self._phase == PHASE_DRAW_TRAIN_CARDS:
            assert 0 <= action < 6, "Improper train card drawing action"

            # Draw a card, action 0-4 indicates a face-up card, 5 indicates a face-down card
            card = self._draw_train_card(action)
            self._player_hands[self._current_player, card] += 1
            self._n_cards_drawn += 1

            if self._n_cards_drawn == 2:
                # Max. no. cards to draw hit, return to the main phase
                terminate = self._end_turn()

                return self._get_obs(), 0, terminate, False, {}
            
            elif (self._card_id_to_color[card] == "gay") and action != 5:
                # Face-up gay card drawn, return to the main phase
                terminate = self._end_turn()

                return self._get_obs(), 0, terminate, False, {}
            
            else:
                # May keep drawing
                return self._get_obs(), 0, False, False, {}
            
        elif self._phase == PHASE_CHOOSE_TICKETS:
            # Choose tickets and add to hand
            chosen_tickets, reward = self._choose_tickets(action)
            self._player_tickets[self._current_player, chosen_tickets] = 1

            assert not np.any(self._player_tickets > 1), "Tickets have been selected multiple times"

            self._pending_tickets[self._current_player] = -1
            terminate = self._end_turn()

            return self._get_obs(), reward, terminate, False, {}
        
        elif self._phase == PHASE_CLAIM_ROUTE:
            assert 0 <= action < len(self._routes), "Improper route claiming action"
            assert self._can_route_be_claimed(action), "Can't afford to claim route"

            self._route_to_claim = action
            reward = self._claim_route(action)

            self._phase = PHASE_CHOOSE_PAYMENT
            self.action_space = self._choose_payment_action_space

            return self._get_obs(), reward, False, False, {}
        
        elif self._phase == PHASE_CHOOSE_PAYMENT:
            assert np.all((self._player_hands[self._current_player] - self._route_cost_payments[action]) >= 0), "Can't afford payment"

            self._pay_for_route(action)
            terminate = self._end_turn()

            return self._get_obs(), 0, terminate, False, {}
        
        else:
            raise RuntimeError(f"Unknown phase {self._phase}")
        
        if self.render_mode == "human":
            self._render_frame()


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed) # Seed self.np_random

        # Initialize state
        self._current_player = 0
        self._phase = PHASE_INITIAL_TICKET_SELECTION
        self.action_space = self._choose_initial_tickets_action_space
        self._initial_ticket_selection_done = np.zeros(self.n_players, dtype=bool)
        self._final_round = False
        self._turns_remaining = 0
        self._train_card_deck = TrainCardDeck(self.np_random)

        self._train_cards = self._train_card_deck.draw(5)
        self._remaining_trains = self._remaining_trains = np.full((self.n_players,), self._n_starting_trains, dtype=int) # Remaining trains back to 40 for each player
        self._routes[:, CLAIMED_BY_INDEX] = -1 # Set all routes to unclaimed
        self._player_tickets = np.zeros((self.n_players, self._n_tickets, 2), dtype=int) # Reset player tickets
        self._route_to_claim = np.array([], dtype=int)

        # Randomly draw 4 train cards and 4 tickets for each player
        for i in range(self.n_players):
            # Draw train cards and add to player
            draws = self._train_card_deck.draw(4)
            np.add.at(self._player_hands[i], draws, 1) 

            # Draw tickets and add to player
            drawn_tickets = self._draw_tickets(4)
            self._pending_tickets[i] = drawn_tickets

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}


    def _get_obs(self):
        return {
            "action_mask": self._get_action_mask(),
            "current_player": self._current_player,
            "pending_tickets": self._pending_tickets[self._current_player],
            "phase": self._phase,
            "player_hand": self._player_hands[self._current_player],
            "player_tickets": np.concatenate([self._player_tickets[self._current_player], self._tickets], axis=1),
            "remaining_trains": self._remaining_trains,
            "routes": self._routes[:, [i for i in range(self._routes.shape[1]) if (i != COST_ID_INDEX)]],
            "train_cards": self._train_cards,
        }


    def _draw_train_card(self, action):
        if action < 5:
            # Face-up card
            drawn_card = self._train_cards[action]
            
            if (np.sum(self._train_card_deck.deck_counts) + np.sum(self._train_card_deck.discard_pile)) > 0:
                # Deck and discard pile aren't empty
                self._train_cards[action] = self._train_card_deck.draw(1)[0] # Draw a new card in place
            else:
                self._train_cards[action] = -1

            return drawn_card
        else:
            # Draw from the deck
            return self._train_card_deck.draw(1)[0]
        


    def _draw_tickets(self, n=3):
        # First check which tickets haven't been drawn yet, then draw 3 randomly
        undrawn_tickets = np.where(~self._player_tickets[:, :, TICKET_OWNED].any(axis=0))[0]

        n_draw = n
        if len(undrawn_tickets) < n:
            # There's not enough tickets left, draw remaining
            n_draw = len(undrawn_tickets)
            
        drawn_tickets = self.np_random.choice(undrawn_tickets, size=n_draw, replace=False)
        
        if n_draw < n:
            # Pad up to n for consistency with observation space
            drawn_tickets = np.pad(drawn_tickets, (0, n - n_draw), mode="constant", constant_values=-1)

        return drawn_tickets
    

    def _choose_tickets(self, action, initial=False):
        # Action indexes choose_ticket_options or initial_choose_ticket_options for a multi binary action
        if initial:
            action = self._choose_initial_ticket_options[action]
        else:
            action = self._choose_ticket_options[action]

        kept_tickets = self._pending_tickets[self._current_player, action]
        reward = 0

        for ticket in kept_tickets:
            reward += self._check_ticket_completion(ticket)
        
        return kept_tickets, reward


    def _check_ticket_completion(self, ticket_id):
        player_graph = self._board_graph.copy() # Copy the base graph
        player_graph.add_edges_from(self._routes[np.where(self._routes[:, CLAIMED_BY_INDEX] == self._current_player)[0], :2].tolist())
        
        # Get info from ticket
        start = self._tickets[ticket_id, 0]
        stop = self._tickets[ticket_id, 1]
        reward = self._tickets[ticket_id, 2]

        if nx.has_path(player_graph, start, stop):
            # Player has completed ticket, return the associated points
            self._player_tickets[self._current_player, ticket_id, 1] = 1 # Set to completed

            return reward
        else:
            # Player has not completed ticket, return 0
            return 0
        

    def _claim_route(self, action):
        self._routes[action, CLAIMED_BY_INDEX] = self._current_player # Updated claimed status
        reward = self._length_to_points[self._routes[action, LENGTH_INDEX]] # Length to points
        self._remaining_trains[self._current_player] -= self._routes[action, LENGTH_INDEX] # Reduce remaining trains
        
        if (self._routes[action, DRAW_BONUS_INDEX] > 0) and ((np.sum(self._train_card_deck.deck_counts) + np.sum(self._train_card_deck.discard_pile)) > 0):
            # Player gets to draw cards form the deck (if there are cards left)
            drawn_cards = self._train_card_deck.draw(self._routes[action, DRAW_BONUS_INDEX])
            self._player_hands[self._current_player, drawn_cards] += 1

        if (self._remaining_trains[self._current_player] <= 2) and (not self._final_round):
            # Player has hit end game condition of two trains left
            self._final_round = True
            self._turns_remaining = self.n_players

        # Check for ticket completion
        uncompleted_owned_tickets = np.where((self._player_tickets[self._current_player, :, TICKET_OWNED] == 1) & (self._player_tickets[self._current_player, :, TICKET_COMPLETED] == 0))[0]
        for ticket_id in uncompleted_owned_tickets:
            reward += self._check_ticket_completion(ticket_id)

        return reward
            

    def _pay_for_route(self, action):
        self._player_hands[self._current_player] -= self._route_cost_payments[action] # Pay the cards, update player hand accordingly
        self._train_card_deck.discard_pile += self._route_cost_payments[action] # Add paid cards to the discard pile
        self._route_to_claim = np.array([])


    def _end_turn(self):
        self._phase = PHASE_MAIN
        self.action_space = self._main_action_space
        self._current_player = (self._current_player + 1) % self.n_players # Next player

        if self._final_round:
            self._turns_remaining -= 1
            if self._turns_remaining == 0:
                return True
            
        return False
    

    # Action legality functions
    def _get_action_mask(self):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action_mask = np.ones(shape=(self.action_space.n,), dtype=np.int8)
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            action_mask = np.ones(shape=self.action_space.shape, dtype=np.int8)

        if self._phase == PHASE_MAIN:
            # Can draw train cards as long as there are some
            if np.all(self._train_cards == -1):
                # All train cards have been drawn
                action_mask[ACTION_DRAW_TRAIN_CARDS] = 0

            # Can draw tickets as long as there's tickets to draw
            if np.sum(self._player_tickets[:, :, TICKET_OWNED].any(axis=0)) == self._n_tickets:
                # All tickets have been drawn
                action_mask[ACTION_DRAW_TICKETS] = 0

            # Whether or not a route can be claimed depends on player hand
            action_mask[ACTION_CLAIM_ROUTE] = 0 # Initialize as unable

            for route in np.where(self._routes[:, CLAIMED_BY_INDEX] == -1)[0]:
                if self._can_route_be_claimed(route):
                    # If a route can be claimed, there exists a valid action, break out of the loop
                    action_mask[ACTION_CLAIM_ROUTE] = 1
                    break 

            # TODO: remove after debugging
            if np.all(action_mask == 0):
                print("ALL ACTIONS UNAVAILABLE")
                print(self._get_obs())
            
        elif self._phase == PHASE_DRAW_TRAIN_CARDS:
            # Can draw cards that aren't set to -1 (in case of empty deck/discard pile)
            action_mask[:5] = (self._train_cards != -1).astype(int)

            if (np.sum(self._train_card_deck.deck_counts) + np.sum(self._train_card_deck.discard_pile)) == 0:
                # Deck is empty, can't draw face-down
                action_mask[5] = 0
            
        elif self._phase == PHASE_CHOOSE_TICKETS:
            # Can draw tickets that aren't set to -1 (in case of empty ticket pile)
            action_mask[:len(self._choose_ticket_options)] = np.array([all((pick == 0 or self._pending_tickets[self._current_player, :3][i] != -1) for i, pick in enumerate(opt)) for opt in self._choose_ticket_options], dtype=np.int8)

        elif self._phase == PHASE_CLAIM_ROUTE:
            # Compute which routes can be claimed     
            action_mask[:] = 0 # Initialize as unable       
            for route in np.where(self._routes[:, CLAIMED_BY_INDEX] == -1)[0]:
                if self._can_route_be_claimed(route):
                    action_mask[route] = 1

        elif self._phase == PHASE_CHOOSE_PAYMENT:
            # Return valid payment options
            action_mask[:] = 0 # Initialize as unable
            cost_id = self._routes[self._route_to_claim, COST_ID_INDEX]
            payment_ids = self._route_cost.loc[cost_id, "payment_ids"]

            for payment_id in payment_ids:
                payment = self._route_cost_payments[payment_id]
                if np.all((self._player_hands[self._current_player] - payment) >= 0):
                    action_mask[payment_id] = 1
        
        if np.sum(action_mask) == 0:
            print(f"Phase: {self._phase}")
            print(f"Action mask: {action_mask}")
            print(f"Player hand: {self._player_hands[self._current_player]}")
            print(f"Train cards: {self._train_cards}")

        assert np.sum(action_mask) > 0, "All actions are masked as 0"

        return action_mask

    
    def _can_route_be_claimed(self, route_id):
        if self._routes[route_id, CLAIMED_BY_INDEX] != -1:
            # Already claimed
            return False
        else:
            # Check if any of the payment options associated with the route can be afforded
            player_hand = self._player_hands[self._current_player, :]
            cost_id = self._routes[route_id, COST_ID_INDEX]
            payment_ids = self._route_cost.loc[cost_id, "payment_ids"]
            payment_options = self._route_cost_payments[payment_ids]

            for payment_option in payment_options:
                if np.all((player_hand - payment_option) >= 0):
                    return True
                
    
    # Rendering functions
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        
# %%

class TrainCardDeck:
    def __init__(self, rng, n_per_color: list[int] = [12, 12, 12, 12, 12, 12, 12, 12, 18]):
        self.n_types = len(n_per_color)
        self.max_counts = np.array(n_per_color)
        self.deck_counts = self.max_counts.copy()
        self.discard_pile = np.zeros_like(self.max_counts)
        self.rng = rng


    def draw(self, n: int) -> np.ndarray:
        if self.deck_counts.sum() < n:
            # If not enough cards are left, reshuffle the deck
            self.reshuffle()

        # Compute drawing probs based on deck and draw n cards
        probs = self.deck_counts / self.deck_counts.sum()
        draws = self.rng.choice(self.n_types, size=n, p=probs)

        for card in draws:
            self.deck_counts[card] -= 1

        return draws
    

    def discard(self, card_ids: np.ndarray):
        # Add card to the discard pile
        for card in card_ids:
            self.discard_pile[card] += 1


    def reshuffle(self):
        self.deck_counts += self.discard_pile
        self.discard_pile[:] = 0
        

# %%
import torch
from torch import nn
import torch.nn.functional as F

class TicketToRideNorthernLightsPPOAgent:
    def __init__(self, env: gym.Env, hidden_dim: int):
        super().__init__()

        self.env = env

        # State encoder for policy network
        self.policy_encoder = nn.Sequential(
            nn.Linear(self.input_dim_from_env(), 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
        )

        # Define the network heads based on action spaces and phases
        self.policy_heads = nn.ModuleDict()
        self.policy_heads[str(PHASE_INITIAL_TICKET_SELECTION)] = nn.Linear(hidden_dim, env._choose_initial_tickets_action_space.n)
        self.policy_heads[str(PHASE_MAIN)] = nn.Linear(hidden_dim, env._main_action_space.n)
        self.policy_heads[str(PHASE_DRAW_TRAIN_CARDS)] = nn.Linear(hidden_dim, env._draw_train_action_space.n)
        self.policy_heads[str(PHASE_CHOOSE_TICKETS)] = nn.Linear(hidden_dim, env._choose_ticket_action_space.n)
        self.policy_heads[str(PHASE_CLAIM_ROUTE)] = nn.Linear(hidden_dim, env._claim_route_action_space.n)
        self.policy_heads[str(PHASE_CHOOSE_PAYMENT)] = nn.Linear(hidden_dim, env._choose_payment_action_space.n)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.input_dim_from_env(), 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        # Initialize optimizers
        self.policy_optimizer = torch.optim.Adam(list(self.policy_encoder.parameters()) + list(self.policy_heads.parameters()))
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()))


    def input_dim_from_env(self):
        input_dim = 0
        for space_name, space in self.env.observation_space.items():
            if space_name not in ["phase", "action_mask"]:
                if isinstance(space, gym.spaces.Discrete):
                    input_dim += 1
                else:
                    input_dim += gym.spaces.utils.flatten_space(space).shape[0]

        return input_dim


    def forward_single(self, state_tensor, phase, action_mask):
        # Get values and policy encodings
        x = self.policy_encoder(state_tensor)
        value = self.value_net(state_tensor)

        mask = torch.tensor(action_mask).bool() # Define boolean mask

        # Pass through policy network with right action head
        logits = self.policy_heads[str(phase)](x) 
        logits_masked = logits.masked_fill(~mask, float("-inf"))
        action_probs = F.softmax(logits_masked, dim=-1)

        return action_probs, value
    

    def forward_batch(self, state_batch, phases_batch, action_masks_batch):
        batch_size = state_batch.shape[0]

        # Get values and policy encodings
        x = self.policy_encoder(state_batch)
        values = self.value_net(state_batch)

        # logits = torch.stack([self.policy_heads[str(phase)](x[i]) for i, phase in enumerate(phases_batch)])
        # mask_tensor = torch.tensor(action_masks_batch, dtype=torch.bool)

        # Iterate over "rows" in batch
        batch_probs = []
        for i in range(batch_size):
            mask = torch.tensor(action_masks_batch[i]).bool() # Define boolean mask

            # Pass through policy network with right action head
            x_single = x[i:i+1]
            logits = self.policy_heads[str(phases_batch[i])](x_single) 
            logits_masked = logits.masked_fill(~mask, float("-inf"))
            action_probs = F.softmax(logits_masked, dim=-1).squeeze(0)
            
            batch_probs.append(action_probs)

        return batch_probs, values
        

    def prepare_state_tensor(self, obs):
        obs_tensors = []
        for key, value in obs.items():
            if key not in ["phase", "action_mask"]:
                if isinstance(value, np.ndarray):
                    tensor = torch.tensor(value.flatten(), dtype=torch.float)
                elif isinstance(value, int):
                    tensor = torch.tensor([value], dtype=torch.float)
                else:
                    print(key, value)

                obs_tensors.append(tensor)
                
        state_tensor = torch.cat(obs_tensors)

        return state_tensor
    

    def unprepare_state_tensor(self, state_tensor):
        obs = {}
        last_index = 0
        for key, value in self.env.observation_space.items():
            if key not in ["phase", "action_mask"]:
                if isinstance(value, gym.spaces.Discrete):
                    size = 1
                    item = state_tensor[last_index:last_index + size].item()
                    
                    # Turn into int if relevant
                    if item % 1 == 0:
                        item = int(item)

                    obs[key] = item

                elif isinstance(value, gym.spaces.Box):
                    size = gym.spaces.flatten_space(value).shape[0]
                    dim  = value.shape
                    item = state_tensor[last_index:last_index + size].reshape(dim).numpy()
                    
                    # If all ints, turn to int array
                    if not np.any(np.vectorize(lambda x: isinstance(x, int))(item)):
                        item = item.astype(int)

                    obs[key] = item

                last_index += size

        assert last_index == len(state_tensor), "Not all observations passed"

        return obs

        
    def get_trajectory_batch(self, n: int = 3, gamma: float = 0.9):
        trajectories = []
        advantages = []

        for _ in range(n):
            obs, _ = self.env.reset()

            trajectory = []
            episode_over = False
            while not episode_over:
                state_tensor = self.prepare_state_tensor(obs)

                action_probs, value = self.forward_single(state_tensor, obs["phase"], obs["action_mask"]) # Get action probabilities from model
                action = action_probs.multinomial(1) # Draw an action
                
                traj_info = [state_tensor, 0, value.detach(), torch.log(action_probs[action]).detach(), action.detach(), obs["phase"], obs["action_mask"]]

                obs, reward, terminated, truncated, _ = self.env.step(action)

                traj_info[REWARD_INDEX] = reward
                trajectory.append(traj_info)
                episode_over = terminated or truncated

            rewards = torch.tensor([traj[REWARD_INDEX] for traj in trajectory])
            values = torch.cat([traj[VALUE_INDEX] for traj in trajectory])
            
            advantages.append(self.compute_advantages(rewards, values, gamma))
            trajectories.append(trajectory)

        advantages = torch.stack([adv for advantage in advantages for adv in advantage]) # Flatten advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Normalize advantages

        return trajectories, advantages
    

    def train(self, n_iterations: int = 100, K: int = 20, n_sample : int = 20, epsilon: float = 0.25):
        policy_losses = []
        value_losses = []
        for _ in tqdm(range(n_iterations)):
            trajectories, advantages = self.get_trajectory_batch(n=n_sample)
            policy_losses_i, value_losses_i = self.optimize(trajectories, advantages, K=K, epsilon=epsilon)
            policy_losses += policy_losses_i
            value_losses += value_losses_i

        fig, axs = plt.subplots(ncols=2)
        axs[0].plot(policy_losses)
        axs[1].plot(value_losses)
        axs[0].set(title="Policy Loss")
        axs[1].set(title="Value Loss")


    def optimize(self, trajectories: list, advantages: list, K: int = 10, batch_size: int = 128, epsilon: float = 0.25):
        states = torch.stack([traj[STATE_TENSOR_INDEX] for trajectory in trajectories for traj in trajectory])
        old_values = torch.stack([traj[VALUE_INDEX] for trajectory in trajectories for traj in trajectory])
        old_log_probs = torch.cat([traj[LOG_ACTION_PROBS_INDEX] for trajectory in trajectories for traj in trajectory])
        actions = torch.cat([traj[ACTION_INDEX] for trajectory in trajectories for traj in trajectory])
        phases = [traj[PHASE_INDEX] for trajectory in trajectories for traj in trajectory]
        action_masks = [traj[ACTION_MASK_INDEX] for trajectory in trajectories for traj in trajectory]

        # Iterate through epochs
        policy_losses = []
        value_losses = []
        for _ in tqdm(range(K)):
            permutation = torch.randperm(states.size()[0]) # Shuffle data

            # Iterate over minibatches
            for i in range(0, states.size()[0], batch_size):
                # Get minibatch
                indices = permutation[i:i+batch_size]
                state_batch = states[indices]

                # Pass data through models
                phases_batch, action_masks_batch = zip(*[(phases[j], action_masks[j]) for j in indices])
                action_probs, values = self.forward_batch(state_batch, phases_batch, action_masks_batch)

                actions_batch = actions[indices]
                new_log_probs = torch.log(torch.stack([action_probs[j][actions_batch[j]] for j in range(len(action_probs))]))

                # Reset optimizer gradients
                policy_loss = self.ppo_loss(new_log_probs, old_log_probs[indices], advantages[indices], epsilon)
                value_loss = F.mse_loss(values, old_values[indices])

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return policy_losses, value_losses


    def compute_advantages(self, rewards, values, gamma: float = 0.9):
        assert len(rewards) == len(values), "Dimensionality mismatch between rewards and values"
        
        n = len(rewards)
        Q_values = []

        # Compute Q value for each action taken
        for i in range(n):
            discount_factors = gamma ** torch.arange(n - i)
            Q_values.append(torch.sum(rewards[i:] * discount_factors))

        # Compute A_t = Q - V
        Q_values = torch.stack(Q_values)
        advantages = Q_values - values

        return advantages
        

    def ppo_loss(self, new_log_probs, old_log_probs, advantages, epsilon: float = 0.2):
        assert len(new_log_probs) == len(old_log_probs), "Dimensionality mismatch between old and new probabilities"

        r = torch.exp(new_log_probs - old_log_probs) # Compute probability ratios
        surr1 = r * advantages  # First part or minimum
        surr2 = torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantages # Second part, clip the ratio
        loss = -torch.min(surr1, surr2).mean() # Take the mean of the minimum between the two, negative to minimize

        return loss
    

    def get_action(self, obs):
        state_tensor = self.prepare_state_tensor(obs)

        logits = self.policy_encoder(state_tensor)

    

#%%

import pickle

env = TicketToRideNorthernLightsEnv()
agent = TicketToRideNorthernLightsPPOAgent(env, hidden_dim=128)
# agent.train(n_iterations=100, K=20)

# with open("trained_agent.pkl", "wb") as f:
#     pickle.dump(agent, f)

# with open("trained_agent.pkl", "rb") as f:
#     agent = pickle.load(f)

trajectories, advantages = agent.get_trajectory_batch(n=1)
trajectory = trajectories[0]

# %%
from vis_constants import cities, routes, player_colors
from vis_classes import TrainCardMarket, PlayerHand, ActionTracker, draw_dashed_line
import pygame
import sys

# Initialize display and clock
pygame.init()

width, height = 700, 700
screen = pygame.display.set_mode((width, height))
screen.fill("white")

pygame.display.set_caption("Ticket to Ride: Northern Lights")
clock = pygame.time.Clock()

# Initialize train cards
card_dim_w = (50, 35)
card_dim_h = (35, 50)
train_cards = [pygame.Surface(card_dim_h) for _ in range(6)]
[train_card.fill("green") for train_card in train_cards[:5]]
train_cards[-1].fill("gray")

# Initialize board
board_image = pygame.image.load("ticket-to-ride-northern-lights.jpg")
board_image = pygame.transform.scale_by(board_image, 1.2)
board_width, board_height = board_image.get_size()
board_loc = ((width - board_width) / 2, (height - board_height) / 2)

[screen.blit(train_card, (width / 2 - 3 * card_dim_h[0] + i * card_dim_h[0], (height - board_height) / 2 - card_dim_h[1])) for i, train_card in enumerate(train_cards)]
screen.blit(board_image, board_loc)

board_outline_out = pygame.Surface((board_width, board_height))
board_outline_out.fill("black")
board_outline_in = pygame.Surface((board_width - 10, board_height - 10))
board_outline_in.fill("white")

screen.blit(board_outline_out, board_loc)
screen.blit(board_outline_in, (board_loc[0] + 5, board_loc[1] + 5))

# Define gameplay objects
train_card_market = TrainCardMarket((width, height), board_height)
player_hands = [PlayerHand((width, height), (board_width, board_height), i) for i in range(agent.env.n_players)]
action_tracker = ActionTracker((500, 30))

# Draw board graph
# for route in routes:
#     pygame.draw.line(screen, colors[route[3]], cities[route[0]][0], cities[route[1]][0], width=3)

[pygame.draw.circle(screen, "black", city_loc, 5) for city_loc, _ in cities]

# Gameplay loop
step_index = 0
is_paused = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_paused = not is_paused

    if not is_paused and step_index < len(trajectory):
        traj = trajectory[step_index]
        obs = agent.unprepare_state_tensor(traj[STATE_TENSOR_INDEX])

        action_tracker.update(traj[PHASE_INDEX], traj[ACTION_INDEX].item(), obs["current_player"])
        action_tracker.draw(screen)

        # Draw a claimed route
        if traj[PHASE_INDEX] == PHASE_CLAIM_ROUTE:
            route = routes[traj[ACTION_INDEX].item()]
            pygame.draw.line(screen, player_colors[obs["current_player"]], cities[route[0]][0], cities[route[1]][0], width=3)
            
        # Update train cards
        train_card_market.update([train_card for train_card in obs["train_cards"].tolist()])
        train_card_market.draw(screen)

        # Update player hands
        player_hands[obs["current_player"]].update(obs["player_hand"])
        [player_hand.draw(screen) for player_hand in player_hands]

        # Draw owned tickets
        player_tickets = obs["player_tickets"]
        player_tickets = player_tickets[np.where(player_tickets[:, 0] == 1)] # Owned tickets
        for ticket in player_tickets:
            draw_dashed_line(screen, player_colors[obs["current_player"]], cities[ticket[2]][0], cities[ticket[3]][0], width=1)

        step_index += 1

        # pygame.image.save(screen, f"frame_{step_index}.png")

    pygame.display.update()
    clock.tick(1)

    if step_index == (len(trajectory) - 1):
        pygame.quit()
        break
        

# %%


