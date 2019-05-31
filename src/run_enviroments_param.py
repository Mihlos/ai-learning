import gym
from gym.spaces import *
import sys

# Pasar como argumento un environment de la lista disponibles.
# Nos devuelve el box de observaciones y acciones disponibles
# y una pequeña ejecución del entorno para empezar a conocerlo.

MAX_EPISODES= 5
MAX_STEPS= 50

def show_spaces(space):
  print(space)
  if isinstance(space, Box):
    print('\nCota inferior(low):\n', space.low)
    print('\nCota superior(high):\n', space.high)

def run_enviroment(argv, env):
  for ep in range(MAX_EPISODES):
    obs = env.reset()
    
    for step in range(MAX_STEPS):
      env.render()
      action = env.action_space.sample()
      obs, reward, done, info= env.step(action)
      if done:
        break
  env.close()

if __name__ == '__main__':
  try:
    env = gym.make(sys.argv[1])
    # Ver el tipo de observaciones, acciones disponibles
    print('Espacio de observaciones:')
    show_spaces(env.observation_space)
    print('Espacio de acciones:')
    show_spaces(env.action_space)
    
    run_enviroment(sys.argv, env)
  except AttributeError:
    pass