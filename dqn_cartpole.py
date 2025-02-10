import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
GAMMA = 0.99  # Discount factor
LEARNING_RATE = 0.001
BATCH_SIZE = 64 # How many per one
MEMORY_SIZE = 10000 # replay buffer
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.999
TRAIN_EVERY = 2


# Neural network (copy-pasted)
# feed-forward, input: state / output: q-value
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Agent
class DQNAgent:
    def __init__(self, state_size, action_size, use_replay=True):
        self.state_size = state_size
        self.action_size = action_size
        self.use_replay = use_replay
        self.epsilon = EPSILON_START
        self.memory = deque(maxlen=MEMORY_SIZE) if use_replay else []
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_tensor)).item()

    def store_experience(self, experience):
        if self.use_replay:
            self.memory.append(experience)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        #current q value / target q-value
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Training Loop
def train_dqn(use_replay=True):
    env = gym.make('CartPole-v1')
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, use_replay)
    rewards_per_episode = []

    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_experience((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)

        #do replay training after every 2nd environment episode
        if use_replay and episode % TRAIN_EVERY == 0:
            agent.train()
            agent.update_target()

        agent.epsilon = max(EPSILON_END, EPSILON_START * (0.99 ** episode))

    env.close()
    return rewards_per_episode


rewards_replay = train_dqn(use_replay=True)
rewards_no_replay = train_dqn(use_replay=False)

plt.plot(rewards_replay, label='With Replay Buffer')
plt.plot(rewards_no_replay, label='Without Replay Buffer')
plt.ylabel('Reward')
plt.legend()
plt.savefig('plot.png')
plt.show()
