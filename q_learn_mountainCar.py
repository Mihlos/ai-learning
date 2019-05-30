import gym

env = gym.make('MountainCar-v0')
#env = gym.make('BipedalWalker-v2')

MAX_EPISODES= 10
#MAX_STEPS= 100

for episode in range(MAX_EPISODES):
  done = False
  total_reward = 0.0
  observation = env.reset()
  step = 0
    
  while not done:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    total_reward += reward
    step +=1
      
  print("\nEpisode {} finished after {} steps. Total Reward: {}".format(episode+1, step, total_reward))
        
env.close()