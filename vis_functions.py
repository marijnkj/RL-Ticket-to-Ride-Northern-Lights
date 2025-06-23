from vis_classes import *
from vis_constants import *
from constants import *
import pygame

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


def draw_base_board(screen, board_image, board_loc, board_outline_out, board_outline_in):
    screen.fill("white")

    # Draw board
    screen.blit(board_image, board_loc)
    screen.blit(board_outline_out, board_loc)
    screen.blit(board_outline_in, (board_loc[0] + 5, board_loc[1] + 5))

    # Draw city circles
    [pygame.draw.circle(screen, "black", city_loc, 5) for city_loc, _ in cities]

    
def draw_frame_info(screen, trajectory, step_index, height, width, auto_play, auto_speed):
    frame_text = font.render(f"Frame: {step_index + 1} / {len(trajectory)}", True, "black")
    screen.blit(frame_text, (10, height - 60))

    controls = [
        "← → : Step back/forward",
        "SPACE: Play/Pause",
        "↑ ↓ : Speed up/down",
        "HOME: Go to start",
        "END: Go to end",
    ]

    for i, control in enumerate(controls):
        control_text = font.render(control, True, "black")
        screen.blit(control_text, (10, height - 40 + i * 15))

    if auto_play:
        auto_text = font.render(f"AUTO (Speed: {auto_speed} fps)", True, "red")
        screen.blit(auto_text, (width - 150, height - 40))
    else:
        manual_text = font.render("MANUAL", True, "blue")
        screen.blit(manual_text, (width - 80, height - 40))


def render_frame(frame_index, trajectory, agent, screen, action_tracker, train_card_market, player_hands, score_board):
    if (frame_index < 0) or (frame_index >= len(trajectory)):
        return
    
    # Get current trajectory step
    traj = trajectory[frame_index]
    obs = agent.unprepare_state_tensor(traj[STATE_TENSOR_INDEX])

    # Update and draw action tracker
    action_tracker.update(traj[PHASE_INDEX], traj[ACTION_INDEX].item(), obs["current_player"])
    action_tracker.draw(screen)

    # Draw the claimed routes up to this point
    for i in range(frame_index + 1):
        past_traj = trajectory[i]

        if past_traj[PHASE_INDEX] == PHASE_CLAIM_ROUTE:
            past_obs = agent.unprepare_state_tensor(past_traj[STATE_TENSOR_INDEX])
            route_index = past_traj[ACTION_INDEX].item()
            route = routes[route_index]

            pygame.draw.line(screen, player_colors[past_obs["current_player"]], cities[route[0]][0], cities[route[1]][0], width=3)

    # Update and draw train card market
    train_card_market.update([train_card for train_card in obs["train_cards"].tolist()])
    train_card_market.draw(screen)

    # Update and draw player hands
    player_hands[obs["current_player"]].update(obs["player_hand"])
    [player_hand.draw(screen) for player_hand in player_hands]

    # Draw owned tickets
    player_tickets = obs["player_tickets"]
    player_tickets = player_tickets[np.where(player_tickets[:, 0] == 1)] # Owned tickets
    for ticket in player_tickets:
        draw_dashed_line(screen, player_colors[obs["current_player"]], cities[ticket[2]][0], cities[ticket[3]][0], width=1)

    # Update and draw score board
    score_board.update(obs["current_player"], int(traj[REWARD_INDEX] * 10))
    score_board.draw(screen)
    
    # Draw frame info
    draw_frame_info()