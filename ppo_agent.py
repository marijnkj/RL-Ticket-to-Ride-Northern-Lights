import gymnasium as gym
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import *

class TicketToRideNorthernLightsPPOAgent:
    def __init__(self, env: gym.Env, policy_lr: float = 3e-4, value_lr: float = 1e-4):
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
        self.policy_heads[str(PHASE_INITIAL_TICKET_SELECTION)] = nn.Linear(128, env._choose_initial_tickets_action_space.n)
        self.policy_heads[str(PHASE_MAIN)] = nn.Linear(128, env._main_action_space.n)
        self.policy_heads[str(PHASE_DRAW_TRAIN_CARDS)] = nn.Linear(128, env._draw_train_action_space.n)
        self.policy_heads[str(PHASE_CHOOSE_TICKETS)] = nn.Linear(128, env._choose_ticket_action_space.n)
        self.policy_heads[str(PHASE_CLAIM_ROUTE)] = nn.Linear(128, env._claim_route_action_space.n)
        self.policy_heads[str(PHASE_CHOOSE_PAYMENT)] = nn.Linear(128, env._choose_payment_action_space.n)

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
        self.policy_optimizer = torch.optim.Adam(list(self.policy_encoder.parameters()) + list(self.policy_heads.parameters()), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), lr=value_lr)


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

        
    def get_trajectory_batch(self, n: int = 3, gamma: float = 0.99, lambd: float = 0.95):
        trajectories = []
        advantages = []
        returns = []
        values = []

        for _ in range(n):
            obs, _ = self.env.reset()

            trajectory = []
            episode_over = False
            while not episode_over:
                state_tensor = self.prepare_state_tensor(obs)

                action_probs, value = self.forward_single(state_tensor, obs["phase"], obs["action_mask"]) # Get action probabilities from model
                action = action_probs.multinomial(1) # Draw an action
                
                traj_info = [state_tensor, 0, value.detach(), torch.log(action_probs[action]).detach(), action.detach(), obs["phase"], obs["action_mask"]]

                phase = obs["phase"]

                obs, reward, terminated, truncated, _ = self.env.step(action)

                if reward != 0:
                    print(f"Reward: {reward}")
                    print(f"Phase: {phase}")
                    print(f"Action: {action.item()}")

                traj_info[REWARD_INDEX] = reward
                trajectory.append(traj_info)
                episode_over = terminated or truncated

            rewards = torch.tensor([traj[REWARD_INDEX] for traj in trajectory])
            values_ep = torch.cat([traj[VALUE_INDEX] for traj in trajectory])
            returns_ep = self.compute_returns(rewards, gamma)

            returns.append(returns_ep)
            values.append(values_ep)

            advantages_ep = self.compute_gae_advantages(rewards, values_ep, gamma, lambd=lambd)
            advantages.append(advantages_ep)

            trajectories.append(trajectory)

        # Flatten and normalize advantages
        advantages = torch.cat(advantages) 
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return trajectories, advantages, returns, values
    

    def compute_returns(self, rewards, gamma: float=0.9):
        n = len(rewards)
        returns = torch.zeros_like(rewards, dtype=torch.float32)

        # Compute returns backwards
        returns[-1] = rewards[-1]
        for i in range(n - 2, -1, -1):
            returns[i] = rewards[i] + gamma * returns[i + 1]

        return returns
    

    def compute_gae_advantages(self, rewards, values, gamma: float=0.99, lambd: float = 0.95):
        n = len(rewards)
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        gae = 0

        # Add a zero at the end for loop below
        values_with_next = torch.cat([values, torch.zeros(1)])

        # Compute GAE backwards
        for i in range(n - 1, -1, -1):
            delta = rewards[i] + gamma * values_with_next[i + 1] - values_with_next[i]
            gae = delta + gamma * lambd * gae
            advantages[i] = gae

        return advantages
    

    def train(self, n_iterations: int = 100, K: int = 20, n_sample : int = 20, gamma: float = 0.9, batch_size: int = 128, epsilon: float = 0.25, entropy_coef: float = 0.01):
        policy_losses = []
        value_losses = []
        all_returns = []
        all_values = []

        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, n_iterations)
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.value_optimizer, n_iterations)

        for i in tqdm(range(n_iterations)):
            trajectories, advantages, returns, values = self.get_trajectory_batch(n=n_sample, gamma=gamma)
            policy_losses_i, value_losses_i = self.optimize(trajectories, advantages, returns, K=K, batch_size=batch_size, epsilon=epsilon, entropy_coef=entropy_coef)
            policy_losses += policy_losses_i
            value_losses += value_losses_i

            # Update learning rate schedulers after each iteration
            self.policy_scheduler.step()
            self.value_scheduler.step()

            all_returns = all_returns + [ret for retur in [retur.tolist() for retur in returns] for ret in retur]
            all_values = all_values + [val for value in [value.tolist() for value in values] for val in value]

            if (i + 1) % 10 == 0:
                fig, axs = plt.subplots(ncols=2)
                axs[0].plot(policy_losses)
                axs[1].plot(value_losses)
                axs[0].set(title="Policy Loss")
                axs[1].set(title="Value Loss")
                fig.suptitle(f"Iteration {i}")
                plt.show()

        print(f"Return stats:")
        print(f"  Mean: {np.mean(all_returns):.4f}")
        print(f"  Std: {np.std(all_returns):.4f}")
        print(f"  Min: {np.min(all_returns):.4f}")
        print(f"  Max: {np.max(all_returns):.4f}")
        print(f"  Non-zero rewards: {np.sum(all_returns != 0)}/{len(all_returns)}")
        
        print(f"Value stats:")
        print(f"  Mean: {np.mean(all_values):.4f}")
        print(f"  Std: {np.std(all_values):.4f}")
        print(f"  Min: {np.min(all_values):.4f}")
        print(f"  Max: {np.max(all_values):.4f}")
        
        if np.std(all_values) < 0.01:
            print("WARNING: Value function may have collapsed!")


    def optimize(self, trajectories: list, advantages: list, returns: list, K: int = 10, batch_size: int = 128, epsilon: float = 0.25, entropy_coef: float = 0.01):
        states = torch.stack([traj[STATE_TENSOR_INDEX] for trajectory in trajectories for traj in trajectory])
        old_log_probs = torch.cat([traj[LOG_ACTION_PROBS_INDEX] for trajectory in trajectories for traj in trajectory])
        actions = torch.cat([traj[ACTION_INDEX] for trajectory in trajectories for traj in trajectory])
        phases = [traj[PHASE_INDEX] for trajectory in trajectories for traj in trajectory]
        action_masks = [traj[ACTION_MASK_INDEX] for trajectory in trajectories for traj in trajectory]

        value_targets = torch.cat(returns)

        # Iterate through epochs
        policy_losses = []
        value_losses = []
        for _ in tqdm(range(K)):
            permutation = torch.randperm(states.size()[0]) # Shuffle data

            # Iterate over minibatches
            for i in range(0, states.size()[0], batch_size):
                # Get minibatch
                indices = permutation[i:i + batch_size]
                state_batch = states[indices]

                # Pass data through models
                phases_batch, action_masks_batch = zip(*[(phases[j], action_masks[j]) for j in indices])
                action_probs, values = self.forward_batch(state_batch, phases_batch, action_masks_batch)

                actions_batch = actions[indices]
                new_log_probs = torch.log(torch.stack([action_probs[j][actions_batch[j]] for j in range(len(action_probs))]))

                # Reset optimizer gradients
                policy_loss = self.ppo_loss(new_log_probs, old_log_probs[indices], action_probs, advantages[indices], epsilon, entropy_coef)
                value_loss = F.mse_loss(values, value_targets[indices].unsqueeze(-1))

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy_encoder.parameters()) + list(self.policy_heads.parameters()), max_norm=0.5)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                self.value_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return policy_losses, value_losses
        

    def ppo_loss(self, new_log_probs, old_log_probs, action_probs, advantages, epsilon: float = 0.2, entropy_coef: float = 0.01):
        assert len(new_log_probs) == len(old_log_probs), "Dimensionality mismatch between old and new probabilities"

        r = torch.exp(new_log_probs - old_log_probs) # Compute probability ratios
        surr1 = r * advantages  # First part or minimum
        surr2 = torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantages # Second part, clip the ratio
        loss = -torch.min(surr1, surr2).mean() # Take the mean of the minimum between the two, negative to minimize

        # Add entropy bonus for exploration
        entropy = torch.stack([-(action_prob * torch.log(action_prob + 1e-8)).sum() for action_prob in action_probs]).mean()
        loss -= entropy_coef * entropy

        return loss