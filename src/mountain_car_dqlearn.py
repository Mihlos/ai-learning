import gym
import numpy as np
import sys
import torch
import random

from libs.decay_schedule import LinearDecaySchedule

from libs.perceptron import SLP

MAX_EPISODES = 50000
STEPS_PER_EPISODE = 200                           # Mountain tiene 200 por defecto.

class QLearn(object):
  def __init__(self, env, learning_rate= 0.05, gamma= 0.98):
    self.obs_shape = env.observation_space.shape
    self.action_shape = env.action_space.n
    
    self.Q = SLP(self.obs_shape, self.action_shape)
    self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)
    self.gamma = gamma
    self.learning_rate = learning_rate

    self.epsilon_max =  1.0
    self.epsilon_min =  0.05
    self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                            final_value= self.epsilon_min,
                                            max_steps = 0.5 * MAX_EPISODES * STEPS_PER_EPISODE)

    self.step_num = 0
    self.policy = self.epsilon_greedy_Q


  def get_action(self, obs):
    return self.policy(obs)
    
  def epsilon_greedy_Q(self, obs): 
    if random.random() < self.epsilon_decay(self.step_num):
      action = random.choice([a for a in range(self.action_shape)])
    else:
      action = np.argmax(self.Q(obs).data.to(torch._C.device('cpu')).numpy())
    return action


  def learn(self, obs, action, reward, next_obs):
    td_target = reward + self.gamma * torch.max(self.Q(next_obs))
    td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
    self.Q_optimizer.zero_grad()
    td_error.backward()
    self.Q_optimizer.step()
    

if __name__ == '__main__':
  env = gym.make('LunarLander-v2')
  agent = QLearn(env)
  first_episode = True
  episode_rewards =  list()
  max_reward = -float('inf')
  for episode in range(MAX_EPISODES):
    done = False
    obs = env.reset()
    total_reward = 0.0
    for step in range(STEPS_PER_EPISODE):
      action = agent.get_action(obs)
      next_obs, reward, done, info = env.step(action)
      agent.learn(obs, action, reward, next_obs)
      obs = next_obs
      total_reward += reward

      if done is True:
        if first_episode:
          max_reward = total_reward
          first_episode = False
        episode_rewards.append(total_reward)
        if total_reward > max_reward:
          max_reward = total_reward
        print('\nEpisodio: {} , Iteraciones: {} , Recompensa: {}, Mejor recompensa: {}'.format(
          episode, step+1 , total_reward, max_reward))
        break
  env.close()
