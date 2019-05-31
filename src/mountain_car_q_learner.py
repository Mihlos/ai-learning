import gym
import numpy as np
import sys

MAX_EPISODES = 50000
STEPS_PER_EPISODE = 200                           # Mountain tiene 200 por defecto.
MAX_NUM_STEPS= MAX_EPISODES * STEPS_PER_EPISODE   # Máximo de steps.
EPSILON_MIN = 0.005                               # Aprendizaje minimo permitido hasta la convergencia del modelo.
EPSILON_DECAY = 250 * EPSILON_MIN / MAX_NUM_STEPS # Caida de epsilon de un paso al siguiente.
ALPHA = 0.05                                      # Ratio de aprendizaje del modelo
GAMMA = 0.98                                      # Factor de descuento del modelo
NUM_DISCRETE_BINS = 30                            # Numero de divisones para discretizar las variables continuas.

class QLearn(object):
  def __init__(self, env):
    self.obs_shape = env.observation_space.shape
    self.obs_low = env.observation_space.low 
    self.obs_high = env.observation_space.high 
    self.obs_bins = NUM_DISCRETE_BINS
    self.obs_width = (self.obs_high-self.obs_low)/self.obs_bins

    self.action_shape = env.action_space.n
    # iniciamos una matriz de 0 para guardar los estados por los que pasa el agente
    # con dimensiones 31 * 31 * 3
    self.Q = np.zeros((self.obs_bins+1, self.obs_bins+1, self.action_shape))
    self.alpha = ALPHA
    self.gamma = GAMMA
    self.epsilon =  1.0

  def discretize(self, obs):                      # Para crear los bins para las medidas.
    # La observación actual menos el minimo del objeto entre el ancho. Truncamos y devolvemos
    # como tupla para tener x e y
    return tuple(((obs-self.obs_low) / self.obs_width).astype(int))


  def get_action(self, obs):
    # Discretizamos la observación que nos devuelve el step
    discrete_obs = self.discretize(obs)
    # Seleccion de la acción que más Epsilon nos dé.
    # Si nuestro epsilon(que inicia en 1) es menor que el minimo permitido para aprender
    # restamos la caida que hemos especificado
    if self.epsilon > EPSILON_MIN:
      self.epsilon -= EPSILON_DECAY
    # Si un numero random entre 0 y 1 es mayor que el epsilon (al principio será muy dificil)
    # retornamos el mayor valor de la matriz del agente
    # en caso contrario estamos iniciando y devolvemos cualquier valor al azar.
    if np.random.random() > self.epsilon:
      return np.argmax(self.Q[discrete_obs])
    else:
      return np.random.choice([a for a in range(self.action_shape)])#Con probabilidad epsilon, elegimos una al azar

  def learn(self, obs, action, reward, next_obs):
    # Discretizamos la observación actual y la que generaremos.
    discrete_obs = self.discretize(obs)
    discrete_next_obs = self.discretize(next_obs)
    
    # Aplicamos la ecuación de Bellman (la he separado para entenderla mejor)
    # Nuestro fin es maximizar la ecuación para obtener cada vez valores mayores.
    
    #td_target = reward + self.gamma * np.max(self.Q[discrete_next_obs])
    #td_error = td_target - self.Q[discrete_obs][action]
    #self.Q[discrete_obs][action] += self.alpha*td_error
    
    # Ecuación completa:
    self.Q[discrete_obs][action] += self.alpha*(reward + self.gamma * np.max(self.Q[discrete_next_obs]) - self.Q[discrete_obs][action])


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
  selection = sys.argv[1]
  
  if selection == 'train':
    learned_policy = train(agent, env)
    np.save('../output/learned_policy', learned_policy)
    launch_agent(agent, env, learned_policy)
  if selection == 'test':
    learned_policy = np.load('../output/learned_policy.npy')
    launch_agent(agent, env, learned_policy)
  else:
    print('Es necesario espeficicar la accion deseada como argumento: train o test')
  
  