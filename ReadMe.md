(en proceso...)
# AI Reinforcement Learning.

Este proyecto es un estudio de como implementar algoritmos de aprendizaje reforzado con técnicas de inteligencia artificial.

El objetivo es crear una inteligencia artificial capaz de solucionar un entorno dado aprendiendo por si misma a solucionar dicho entorno y conseguir un objetivo determinado.

Las herramientas usadas son:
* Python
* OpenAI Gym
* Pytorch

## Estructura de carpetas

* aux.
* media.
  * Bipedal
  * Mountain
* output.
  * models
    * Bipedal
* presentation.
  * openai-docker.
* src
  * libs


#### aux
Ha servido como directorio de pruebas. Actualmente solo contiene un fichero que se usó para aprendizaje y pruebas.

#### media
Visualizaciones generadas por los entornos. Cada vez que se ejecuta un entorno estas carpetas se sobre escriben.

#### output 
Contiene ficheros generados por la ejecución de los scripts. 
En models están los modelos guardados con diferente nivel de entrenamiento.
El archivo learned_policy.npy es una matriz multidimensional numpy que contiene la solución para resolver el entorno MountainCar.

#### presentation
Contiene un directorio ejemplo de como encapsular el proyecto en dockers.
Se ha decidido no incluir la imagen completa ya que no podemos renderizar desde docker.
Tambien encontramos el video de BipedalWalker a modo de presentación.

#### src
Los .py para ejecutar los diferentes programas. Especificaremos más sobre ellos.
La carpeta libs contiene librerias creadas que son utilizadas en los diferentes ficheros de ejecución.

## Proceso de aprendizaje y diferentes scripts.

El proceso de aprendizaje se basa en la ecuación de Bellman y el principio de incertidumbre de Markov.

Implementando de diferentes metodos este mismo principio conseguimos aprendizaje reforzado cada vez más potente.

Durante el preoceso de desarrollo de este proyecto he ido pasando por diferentes etapas. A cada paso el mayor conocimiento que iba adquiriendo me permitia implementar algoritmos mas potentes hasta llegar a redes neuronales para solucionar las diferentes partes de la ecuación.

### 1. run_environments_params.py
En este scrip se empieza a tratar la libreria Gym de OpenAI, permite ejecutar diferentes entornos pasados por parametros y nos devuelve la información de el ambito de obeservacione y acciones disponibles para el entorno pedido.
Es un excelente ejemplo para empezar a enteder la estructura de accion-reconpensa-entorno de la librería.

### 2. taxi_reinforce.py
Primer entorno solucionado para empezar a entender y aplicar la ecuación de Bellman. Como maximizar la ecuación para obtener un resultado deseado.

### 3. mountain_car_qlearn.py
Siguiente entorno solucionado separando el algoritmo de aprendizaje a una librería. Además incluye la discretizacion de las variables de observación para poder procesarlas.
Se implementa el principio de incertidumbre de Markov para la toma de decisiones.
El agente no solo busca el objetivo a toda consta. Tiene un elenco de acciones disponibles y toma la decisión en función a ellas incluyendo un elemento aleatorio.

### 4. bipedal_walker_ddpg.py
El entorno objetivo que deseaba solucionar. Todos los anteriores scripts han sido un entrenamiento que me han llevado a tener la capacidad de comprensión y abstracción necesarias para solucionar un problema mucho más complejo como el que presenta este entorno.

El algoritmo usado es DDPG (Deep Deterministic Policy Gradient). 
https://spinningup.openai.com/en/latest/algorithms/ddpg.html
https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

DDPG puede ser solo utilizado para entornos con observaciones y acciones continuas.
Es ideal cuando hay que evaluar continuamente un entorno y tomar decisiones en función a la observación.
También utiliza un buffer de recarga para "samplear" la experiencia que usamos para actualizar las redes neuronales.
