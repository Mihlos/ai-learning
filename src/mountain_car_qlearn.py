import gym
import numpy as np
import sys

from libs.Qlearn import QLearn

MAX_EPISODES = 50000

# Función para entrenar al agente.
def train(agent, env):
  best_reward = -float('inf')
  for episode in range(MAX_EPISODES):
    done = False
    total_reward = 0.0
    obs = env.reset()
    while not done:
      action = agent.get_action(obs) # Acción elegida según la ecuación de Q-LEarning
      # step devuelve 4 valores, hay que espeficiar los 4 si no da error.
      next_obs, reward, done, info = env.step(action)
      agent.learn(obs, action, reward, next_obs)
      obs = next_obs
      total_reward += reward
    if total_reward > best_reward:
      best_reward = total_reward
    print("Episode: {} Reward: {} Best Reward: {} Epsilon: {} "
          .format(episode+1, total_reward, best_reward, agent.epsilon )) 
  # Devolvemos el mejor valor de la matriz aprendida, el eje 2 
  # El 0 y 1 son los valores de entorno y aceleración. El 2 los movimientos.
  return np.argmax(agent.Q, axis = 2)

# Función para testear lo aprendido.
def test(agent, env, policy):
  done = False
  total_reward = 0.0
  obs = env.reset()
  while not done:
    #env.render()
    action = policy[agent.discretize(obs)]
    next_obs, reward, done, info = env.step(action)
    obs = next_obs
    total_reward += reward
  return total_reward

# Metodo de grabación para evaluar el agente
# Con el wrappers no hace falta lanzar el env.render()
def launch_agent(agent, env, learned_policy):
  monitor_path = '../media'
  env = gym.wrappers.Monitor(env, monitor_path, video_callable=lambda episode_id: True, force = True)
  for i in range(10):
    test(agent, env, learned_policy)
  env.close()

if __name__ == '__main__':
  env = gym.make('MountainCar-v0')
  agent = QLearn(env)
  # Pasamos por param. si queremos train o test.
  selection = sys.argv[1]
  
  if selection == 'train':
    learned_policy = train(agent, env)
    np.save('../output/learned_policy', learned_policy)
    launch_agent(agent, env, learned_policy)
  elif selection == 'test':
    learned_policy = np.load('../output/learned_policy.npy')
    launch_agent(agent, env, learned_policy)
  else:
    print('Es necesario espeficicar la accion deseada como argumento: train o test')
  
  