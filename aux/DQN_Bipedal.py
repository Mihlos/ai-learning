#import pandas
#import keras

# Para trabajar con GPU
from keras import backend as K

from keras.models import load_model
from keras.initializers import RandomUniform
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import time
import numpy as np
import gym
import random
from collections import deque
import tensorflow as tf
import os
    
class AGENT:
    def __init__(self, obs_len, act_len, lr, gamma, s_link, l_link=''):
        self.obs_len = obs_len  #  Observation array length(24)   obs_len = env.observation_space.shape[0]
        self.act_len = act_len  #   Action space length(4)       act_len = env.action_space.shape[0]
        '''
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = .05
        '''
        self.gamma = gamma
        self.lr = lr

        self.l_link = l_link # Ruta de fichero para cargar modelo.
        self.s_link = s_link # Ruta de salvado del modelo.

        self.deck = deque(maxlen=2000)
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.weight_decay = 10**-2
        self.los = []
        
        # Si hay un archivo igual a la ruta de salvado lo cargamos.
        # Tambien podemos especificarlo con el parámetro nombrado 'l_link'
        if os.path.isfile('./' + self.s_link):
            print("CARGANDO MODELO EXISTENTE...")
            self.model = load_model(self.s_link)
            print(self.model.summary())
        else:
            # Llamada a la funcion MODEL para iniciar un modelo para el agente.   
            print("CREANDO UN NUEVO MODELO...")
            self.model = self.MODEL()      
        
        # Valores iniciales de observacion, recompensa, accion...
        self.ep_obs, self.ep_rewards, self.ep_action, self.ep_obs_new, self.ep_flags = [], [], [], [], []
        
    # Elegir la accion a realizar para el paso.
    # Si un número random es OJO CON ESTO...        
    def choose_action(self,observation):
        if np.random.rand() <= self.eps : 
            action = np.random.uniform(-1,1,4)
        else:    
            probs = self.model.predict(observation)    
            #action = np.argmax(probs[0])
            action = probs[0]
        return action
                
    # Función para guardar cada step realizado.
    def storing(self, observation, action, reward, observation_new, flags ):
        self.deck.append((observation, action, reward, observation_new, flags ))
        self.ep_rewards.append(reward)
        
    # Salvar el modelo si es exitoso.    
    def save(self,name):
        self.model.save(name)

    # Modelo
    def MODEL(self):                     
        model = Sequential()
        #linea añadida
        model.add(Dense(48, input_dim = self.obs_len, activation="relu"))
        #cambiado el input_dim a borrado, era el de arriba.
        model.add(Dense(96, activation='relu' , kernel_regularizer = l2(self.weight_decay)))
        model.add(Dense(192,  activation = 'relu', kernel_regularizer = l2(self.weight_decay)))
        model.add(Dense(96,  activation = 'relu', kernel_regularizer = l2(self.weight_decay)))
        model.add(Dense(self.act_len, activation = 'tanh',\
                        bias_initializer = RandomUniform(minval =- 0.003, maxval = 0.003)))
        model.compile(loss = 'mse', optimizer = Adam(lr = self.lr))
                    
        return model

    # Metodo de entrenamiento eligiendo un rango de osbservaciones guardadas
    # en el deck.    
    def TRAIN(self, batch):
        sample_indx = random.sample(self.deck, batch)
        self.los = []
        
        for observation, act, reward, obs_new, done in sample_indx:            
            target = np.repeat(reward,4).reshape(1,-1)
            if not done: #((1-ALPHA)*xreward)+ (ALPHA* (GAMMA * futurereward))
                target = ( (1.0-0.1)*reward + 0.1 * (self.gamma*self.model.predict(obs_new)[0]))                
                target = target.reshape(1,-1)
            #target_old = self.model.predict(observation)
            #target_old[0][act] = target
            target_old = target
            # Train
            #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_‌​parallelism_threads=‌​32, inter_op_parallelism_threads=32)))
            history = self.model.fit(x=observation, y=target_old,\
                                #batch_size=1,\
                                verbose=0,\
                                epochs=5)
            self.los.append(history.history['loss'])    
            
            self.ep_obs, self.ep_rewards, self.ep_action, self.ep_obs_new, self.ep_flags = [], [], [], [], []
        
        mm = np.mean(self.los)                
        if self.eps >= self.eps_min:
            self.eps *= self.eps_decay
        #self.save(self.s_link)
        return history, mm

if __name__ == '__main__':
    
    BATCH = 256 # Ancho de la muestra a elegir para pasar al modelo y evaluar
    rendering = input("Visualize rendering ? [y/n]:  ")
    
    s_link = "BipedalWalker_model.h5"
    
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True  #flag for rendering the environment
    EPISODES = 9000    # Number of episodes
    
    env = gym.make('BipedalWalker-v2')
    env = env.unwrapped
    

    # Observation and Action array length
    obs_len = env.observation_space.shape[0] 
    act_len = env.action_space.shape[0]
    lr = 0.0001
    gamma = 0.99
    agent = AGENT(obs_len, act_len, lr, gamma, s_link)
    
    rewards_over_time = []
    error = []
    epsilon = []
    rew_var = []
    rew_mean = []
    mean_100 = []

    seed = np.random.seed(1986)
    win = 0
        
    # Start running the episodes        
    for i in range(EPISODES): 
        observation = env.reset()         
        observation = observation.reshape(1,-1)                
        start = time.time()
        counter = 0
        while True:            
            if RENDER_ENV:
                env.render()
            
            action = agent.choose_action(observation)
            observation_new, reward, done, inf = env.step(action)
            observation_new = observation_new.reshape(1,-1)                    
            counter +=1

            # Store new information
            agent.storing(observation, action, reward, observation_new, done)   
            observation = observation_new   

            # Measure the time
            end = time.time()
            time_space = end - start
            
            if time_space > 10:
                done = True
          
            # Sum the episode rewards
            ep_rew_total = sum(agent.ep_rewards)
            mean = np.mean(agent.ep_rewards)
            var = np.var(agent.ep_rewards)
            if ep_rew_total < -300:
                done = True
            
            if done == True:
                rewards_over_time.append(ep_rew_total)
                rew_mean.append(mean)
                rew_var.append(var)
                max_reward = np.max(rewards_over_time)
                episode_max = np.argmax(rewards_over_time)
                
                if ep_rew_total >=300 :
                    win = win + 1
                    agent.save(s_link)
                                        
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("Episode: ", i)
                print("Time: ", np.round(time_space, 2),"secs")
                print("Traj: " + str(counter))
                print("Reward:", ep_rew_total)
                print("Maximum Reward: " + str(max_reward) + "  on Episode: " + str(episode_max))
                print("Times win: " + str(win))
                
                if i % 100 ==0:
                    print("Mean reward of the past 100 episodes: ", str(np.mean(rewards_over_time[-100:])))
                    mean_100.append(np.mean(rewards_over_time[-100:]))
                    f = open('results.txt','a')
                    f.write('\n' + str(np.mean(rewards_over_time[-100:])))
                    f.close()
                
                # Start training the Neural Netwinork
                #if BATCH >= len() 
                hist, mm= agent.TRAIN(BATCH)
                
                epsilon.append(agent.eps)
                                           
                error.append(mm)
                
                if max_reward > RENDER_REWARD_MIN: RENDER_ENV = True
                
                break
    
    np.save("rewards_over_time", rewards_over_time)
    np.save("mean100", mean_100) 
            
    plt.figure(figsize=(8,6))
    plt.plot(error)
    plt.xlabel("Episodes")
    plt.ylabel("Average Error")
    plt.title("Average_Loss Vs Episodes")
    plt.show()
    plt.savefig("Average_Loss_Vs_Episodes.png")
    
    plt.figure(figsize=(8,6))
    plt.plot(epsilon)
    plt.xlabel("Episodes")
    plt.ylabel("Epsilon value")
    plt.title("Epsilon Vs Episodes")
    plt.show()
                
    plt.figure(figsize=(8,6))            
    plt.plot(rewards_over_time, label="Rewinards")
    plt.plot(rew_mean, label="Mean")
    plt.plot(rew_var, label="Variance")    
    plt.xlabel("Episodes")
    plt.ylabel("Rewinards")
    plt.title("Rewinards per Episode")
    plt.legend(loc=0)
    plt.show()        
    plt.savefig("Rewinards_per_Episode.png")        
            
    plt.figure(figsize=(8,6))
    plt.plot(mean_100)
    plt.xlabel("100_episodes")
    plt.ylabel("Mean_value")
    plt.xticks(np.arange(0,9000,100))
    plt.savefig("mean_100.png") 