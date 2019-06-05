import gym
import torch
import numpy as np
import sys
import os
import argparse

import matplotlib.pyplot as plt

from libs.ddpg_agent import Agent

parser = argparse.ArgumentParser(description='''
Script para ejecutar el entorno BipedalWalker de OpenAI Gym.
Permite entrenar al agente guardando el modelo.
Permite ver y guardar en .mp4 las ejecuciones de un modelo.

Si elegimos un agente entrenado hay que especificar la ruta
a los archivos elegidos.''')

parser.add_argument(
    'mode',
    type=str,
    help='Modo de ejecución. Elegir entre train o test.'
)
parser.add_argument(
    'trained',
    type=str,
    choices=['yes', 'no'],
    help='Elegir si el agente tendrá entrenamiento previo o no. [yes/no]'
)
parser.add_argument(
    '-l',
    '--load',
    type=str,
    help='Ruta para cargar el modelo entrenado.'
)
parser.add_argument(
    '-e',
    '--episodes',
    default = 10,
    type=int,
    help='Número de episodios a realizar.'
)
args = parser.parse_args()

# Argumentos de elección
mode = args.mode
trained = args.trained
if trained == 'yes': trained = 1
else: trained = 0
load_path = args.load
episodes = args.episodes

# Parámetros del entorno
env = gym.make('BipedalWalker-v2')
obs_len = env.observation_space.shape[0]
act_len = env.action_space.shape[0]
step = 2000
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
            for _ in range(self.step):
                action = agent.act(obs, self.noise)
                next_state, reward, done, info = self.env.step(action[0])
                #Actualiza el aprendizaje del agente.
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
        self.env = gym.wrappers.Monitor(self.env, monitor_path, video_callable=lambda episode_id: True, force = True)
        for i in range(self.episodes):
            obs = self.env.reset()
            score = 0
            for _ in range(self.step): 
                action = agent.act(obs, self.noise)
                next_state, reward, done, info = self.env.step(action[0])
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
    