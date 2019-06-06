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

