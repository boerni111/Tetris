import gymnasium as gym
import torch
from src.agent import DQNAgent
from src.utils import pad_to_target
from tetris_gymnasium.envs.tetris import Tetris

# Hyperparameters
batch_size = 32
episodes = 500
learning_rate = 0.001
gamma = 0.99
epsilon = 0.6
epsilon_decay = 0.99995
buffer_size = 10000


def train():
    env = gym.make("tetris_gymnasium/Tetris", render_mode="ansi")
    state_dim = torch.empty((4,24,18))
    agent = DQNAgent(state_dim,env.action_space.n,learning_rate,gamma,epsilon,epsilon_decay,buffer_size)

    for i in range(episodes):
        state = env.reset()[0]
        current_image_curr = torch.tensor(state["board"])
        current_queue_curr = pad_to_target(torch.tensor(state["queue"]))
        current_replace_item_curr = pad_to_target(torch.tensor(state["holder"]))
        current_mask_curr = torch.tensor(state["active_tetromino_mask"])
        stacked_tensor_curr = torch.stack([current_image_curr,current_queue_curr,current_replace_item_curr,current_mask_curr], dim=0)  # Form: (4, 24, 18)

        done = False
        total_reward = 0
        while not done:
            action = agent.act(stacked_tensor_curr)
            next_state,reward,done,truncated, info = env.step(action)
            lines_cleared = info["lines_cleared"]
            curr_reward = 0
            if lines_cleared > 0:
                curr_reward = 3
            reward = curr_reward + 0.01
            if done:
                reward = -1

            current_image = torch.tensor(next_state["board"])
            current_queue = pad_to_target(torch.tensor(next_state["queue"]))
            current_replace_item = pad_to_target(torch.tensor(next_state["holder"]))
            current_mask = torch.tensor(next_state["active_tetromino_mask"])
            stacked_tensor = torch.stack([current_image,current_queue,current_replace_item,current_mask], dim=0)  # Form: (4, 24, 18)

            agent.remember(stacked_tensor_curr,action,reward,stacked_tensor,done)
        
            stacked_tensor_curr = stacked_tensor
            total_reward += reward

            agent.replay(batch_size)

        if i % 50 == 0:
            print(f"Episode {i+1}:{info} Total_reward:{total_reward} Epsilon:{agent.epsilon}")

    # Save the trained model
    torch.save(agent.model.state_dict(), "tetris_model.pth")

if __name__ == "__main__":
    train()