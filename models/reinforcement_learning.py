import torch
import torch.nn as nn
import torch.optim as optim
import gym

class HedgingAgent(nn.Module):
    """
    A simple reinforcement learning agent for option hedging using PyTorch.
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the hedging agent.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space (e.g., number of
                shares to buy/sell).
        """
        super(HedgingAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        """
        Computes the action given the state.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The action to take.
        """
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        action = self.fc3(x)
        return action

def train_hedging_agent(env, agent, num_episodes=1000, learning_rate=0.001):
    """
    Trains the reinforcement learning agent.

    Args:
        env (gym.Env): The environment (simulating option dynamics).
        agent (nn.Module): The hedging agent.
        num_episodes (int): The number of training episodes.
        learning_rate (float): The learning rate.

    Returns:
        list: A list of episode rewards.
    """
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state)
            action = agent(state_tensor)
            #  Implement a policy (e.g., epsilon-greedy) to select action
            epsilon = 0.1
            if torch.rand(1) < epsilon:
                action = torch.randn_like(action) # Explore
            next_state, reward, done, _ = env.step(action.detach().numpy())  # .detach()
            reward_tensor = torch.FloatTensor([reward])
            # Store the transition in memory
            next_state_tensor = torch.FloatTensor(next_state) # Cast to tensor
            # Compute the loss (e.g., using TD error)
            target = reward_tensor + 0.99 * agent(next_state_tensor).max() * (1 - done) # changed arguments
            prediction = agent(state_tensor).gather(0, action.argmax(0).view(-1)) # changed arguments
            loss = (prediction - target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    return episode_rewards