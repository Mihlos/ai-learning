import gym
import torch
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

from libs.ddpg_agent import Agent

# Parámetros de elección
mode = sys.argv[1]
load_path = sys.argv[2] if len(sys.argv) == 3 else ''

# Parámetros del entorno
env = gym.make('BipedalWalker-v2')
#env = gym.make('BipedalWalkerHardcore-v2')

obs_len = env.observation_space.shape[0]
act_len = env.action_space.shape[0]
episodes = 3000
step = 2000
trained = 1
noise = 0

# Creación del agente
agent = Agent(state_size=obs_len, action_size=act_len, random_seed=0)

class Ignition():
    def __init__(self, env, episodes, step, noise, load_path=''):
        self.episodes = episodes
        self.step = step
        self.noise = noise
        self.load = load_path
        self.reward_list = []

        self.env = env

    def load_train(self):
        print('CARGANDO MODELOS...')
        agent.actor_local.load_state_dict(torch.load(os.path.join(self.load ,'checkpoint_actor.pth'), map_location="cpu"))
        agent.critic_local.load_state_dict(torch.load(os.path.join(self.load ,'checkpoint_critic.pth'), map_location="cpu"))
        agent.actor_target.load_state_dict(torch.load(os.path.join(self.load ,'checkpoint_actor.pth'), map_location="cpu"))
        agent.critic_target.load_state_dict(torch.load(os.path.join(self.load ,'checkpoint_critic.pth'), map_location="cpu"))

    def train(self):
        self.reward_list = []
        for i in range(self.episodes):
            obs = self.env.reset()
            score = 0
            for t in range(self.step):
                action = agent.act(obs, self.noise)
                next_state, reward, done, info = env.step(action[0])
                agent.step(obs, action, reward, next_state, done)
                # squeeze elimina una dimension del array.
                obs = next_state.squeeze()
                score += reward

                if done:
                    print('Reward: {} | Episode: {}/{}'.format(score, i, self.episodes))
                    break

            self.reward_list.append(score)

            if score >= 250:
                print('Task Solved')
                torch.save(agent.actor_local.state_dict(), os.path.join('../output/models/Bipedal',  'checkpoint_actor_solved'+str(score)+'.pth'))
                torch.save(agent.critic_local.state_dict(), os.path.join('../output/models/Bipedal',  'checkpoint_critic_solved'+str(score)+'.pth'))   
                torch.save(agent.actor_target.state_dict(), os.path.join('../output/models/Bipedal',  'checkpoint_actor_t_solved'+str(score)+'.pth'))
                torch.save(agent.critic_target.state_dict(), os.path.join('../output/models/Bipedal',  'checkpoint_critic_t_solved'+str(score)+'.pth'))

        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_t.pth')
        torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_t.pth')

        print('Training saved...')
        return self.reward_list

    def test(self):
        monitor_path = '../media/Bipedal'
        #self.env = gym.wrappers.Monitor(self.env, monitor_path, video_callable=lambda episode_id: True, force = True)
        for i in range(self.episodes):
            obs = self.env.reset()
            score = 0
            for t in range(self.step):
                env.render()
                action = agent.act(obs, self.noise)
                next_state, reward, done, info = env.step(action[0])
                # squeeze elimina una dimension del array.
                obs = next_state.squeeze()
                score += reward

                if done:
                    print('Reward: {} | Episode: {}/{}'.format(score, i, self.episodes))
                    break

    def plot_scores(self):
        fig = plt.figure()
        plt.plot(np.arange(1, len(self.reward_list) + 1), self.reward_list)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

if __name__=='__main__':
    robot = Ignition(env, episodes, step, noise, load_path)
    if trained:
        robot.load_train()
    if mode == 'train':
        robot.train()
        robot.plot_scores()
    if mode == 'test':
        robot.test()
    
    env.close()
    