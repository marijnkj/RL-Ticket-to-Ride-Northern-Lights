#%%
from environment import TicketToRideNorthernLightsEnv
from ppo_agent import TicketToRideNorthernLightsPPOAgent
import pickle

env = TicketToRideNorthernLightsEnv()
agent = TicketToRideNorthernLightsPPOAgent(env, policy_lr=3e-3, value_lr=3e-4)
agent.train(n_iterations=10, K=4, n_sample=20, gamma=0.99, batch_size=256, entropy_coef=0.1)

with open("trained_agent.pkl", "wb") as f:
    pickle.dump(agent, f)

# with open("trained_agent_small.pkl", "rb") as f:
#     agent = pickle.load(f)

trajectories, advantages, rewards, values = agent.get_trajectory_batch(n=1)
trajectory = trajectories[0]

# %%
from vis_classes import *
from vis_functions import *
import pygame
import sys

# Initialize display and clock
pygame.init()

width, height = 700, 700
screen = pygame.display.set_mode((width, height))
screen.fill("white")

pygame.display.set_caption("Ticket to Ride: Northern Lights")
clock = pygame.time.Clock()

# Initialize board
board_image = pygame.image.load("ticket-to-ride-northern-lights.jpg")
board_image = pygame.transform.scale_by(board_image, 1.2)
board_width, board_height = board_image.get_size()
board_loc = ((width - board_width) / 2, (height - board_height) / 2)

board_outline_out = pygame.Surface((board_width, board_height))
board_outline_out.fill("black")
board_outline_in = pygame.Surface((board_width - 10, board_height - 10))
board_outline_in.fill("white")

# Define gameplay objects
train_card_market = TrainCardMarket((width, height), board_height)
player_hands = [PlayerHand((width, height), (board_width, board_height), i) for i in range(agent.env.n_players)]
action_tracker = ActionTracker((500, 30))
score_board = ScoreBoard(agent.env.n_players, (10, 10), len(trajectory))

# Gameplay loop
step_index = 0
is_paused = False
auto_play = True
auto_speed = 1
running = True

draw_base_board(screen, board_image, board_loc, board_outline_out, board_outline_in)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Toggle auto-play
                auto_play = not auto_play
                is_paused = not auto_play
                
            elif event.key == pygame.K_LEFT:
                # Step backward
                if step_index > 0:
                    step_index -= 1
                auto_play = False
                is_paused = True
                
            elif event.key == pygame.K_RIGHT:
                # Step forward
                if step_index < len(trajectory) - 1:
                    step_index += 1
                auto_play = False
                is_paused = True
                
            elif event.key == pygame.K_UP:
                # Increase auto-play speed
                auto_speed = min(auto_speed + 1, 10)
                
            elif event.key == pygame.K_DOWN:
                # Decrease auto-play speed
                auto_speed = max(auto_speed - 1, 1)
                
            elif event.key == pygame.K_HOME:
                # Go to beginning
                step_index = 0
                auto_play = False
                is_paused = True
                
            elif event.key == pygame.K_ESCAPE:
                # Quit
                running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                is_paused = not is_paused

    # Auto-play
    if auto_play and not is_paused:
        if step_index < len(trajectory) - 1:
            # Only continue if not reached end
            step_index += 1
        else:
            # Reached end, pause auto-play
            auto_play = False
            is_paused = True

    render_frame(step_index, trajectory, agent, screen, action_tracker, train_card_market, player_hands, score_board, height, width, auto_play, auto_speed, board_image, board_loc, board_outline_out, board_outline_in)

    pygame.display.update()

    # Control frame rate
    if auto_play:
        clock.tick(auto_speed)
    else:
        clock.tick(60)

pygame.quit()
sys.exit()
