import gymnasium as gym
import torch
import imageio
from src.agent import DQNAgent
from src.utils import pad_to_target
from tetris_gymnasium.envs.tetris import Tetris


def play_and_record():
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    state_dim = torch.empty((4,24,18))
    agent = DQNAgent(state_dim,env.action_space.n,learning_rate=0,gamma=0,epsilon=0,epsilon_decay=0,buffer_size=0)
    agent.model.load_state_dict(torch.load("tetris_model.pth"))

    frames = []
    state = env.reset()[0]
    current_image_curr = torch.tensor(state["board"])
    current_queue_curr = pad_to_target(torch.tensor(state["queue"]))
    current_replace_item_curr = pad_to_target(torch.tensor(state["holder"]))
    current_mask_curr = torch.tensor(state["active_tetromino_mask"])
    stacked_tensor_curr = torch.stack([current_image_curr,current_queue_curr,current_replace_item_curr,current_mask_curr], dim=0)  # Form: (4, 24, 18)

    terminated = False
    while not terminated:
        frames.append(env.render())
        action = agent.act(stacked_tensor_curr)
        observation, reward, terminated, truncated, info = env.step(action)
        current_image = torch.tensor(observation["board"])
        current_queue = pad_to_target(torch.tensor(observation["queue"]))
        current_replace_item = pad_to_target(torch.tensor(observation["holder"]))
        current_mask = torch.tensor(observation["active_tetromino_mask"])
        stacked_tensor = torch.stack([current_image,current_queue,current_replace_item,current_mask], dim=0)  # Form: (4, 24, 18)

        stacked_tensor_curr = stacked_tensor

    env.close()
    imageio.mimsave('tetris.gif', frames, fps=10)

if __name__ == "__main__":
    play_and_record()