import gym
import time
import numpy as np
from google_speech import Speech

env = gym.make('BipedalWalker-v2')

expresiones = ['aiba', 'la virgen', 'pero eres tonto', 'no me levanto', 'menuda hostia']
MAX_EPISODES= 10
MAX_STEPS= 500

for episode in range(MAX_EPISODES):
    observation= env.reset()
    
    for step in range(MAX_STEPS):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # observation o next_state
        #print("\n\n",observation)

        if done:
            # La chorrada number one
            text = np.random.choice(expresiones)
            speech = Speech(text,'es')
            sox_effects = ("speed", "1.5")
            speech.play(sox_effects)
            time.sleep(1.0)    # pause 1 second
            # Fin de la chorrada
            print("\nEpisode {} finished after {} timesteps".format(episode+1, step))
            break

env.close()