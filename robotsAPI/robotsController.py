# Model design
from owlready2 import *
import agentpy as ap
import numpy as np

# Visualization
import seaborn as sns

#Pathfinding
import math
import heapq

#Misc
from matplotlib import pyplot as plt
import IPython
import random

def heuristic(a,b):
  #Distancia de Manhattan, resta el valor absoluto de la XY axtual con la XY destino
  # a[0] = x inicial
  # b[0] = x final
  # a[1] = y inicial
  # b[1] = y final
  return abs(a[0] - b[0]) + abs(a[1] - b[1])

"""
def a_star(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))

    # Mapeo del camino
    came_from = {}

    # Generar todas las posiciones posibles dentro del grid
    all_positions = [(x, y) for x in range(grid.shape[0]) for y in range(grid.shape[1])]

    # Inicializar g_score y f_score
    g_score = {node: float("inf") for node in all_positions}
    g_score[start] = 0

    f_score = {node: float("inf") for node in all_positions}
    f_score[start] = heuristic(start, goal)

    while open_list:
        current = heapq.heappop(open_list)[1]

        # Si se alcanza el objetivo, reconstruir y devolver el camino
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Devolver el camino invertido

        # Explorar vecinos
        for neighbor in get_neighbors(grid, current):
            tentative_g_score = g_score[current] + 1  # Suponiendo un costo uniforme

            if tentative_g_score < g_score[neighbor]:
                # Este camino es mejor que el anterior, actualizamos
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                # Añadir a la lista de exploración
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []  # No se encontró camino
"""

def get_neighbors(grid,node):
  neighbors = []

  x,y = node

  if x > 0:
    neighbors.append((x-1,y))

  if x < grid.shape[1] - 1:
    neighbors.append((x+1,y))

  if y > 0:
    neighbors.append((x,y-1))

  if y < grid.shape[0] -1:
    neighbors.append((x,y+1))

  return neighbors


#Creamos la ontologia
onto = get_ontology("file://ontologia.owl")

with onto:
  class Entity(Thing):
    pass

  class Robot(Entity):
    pass

  class Container(Entity):
    pass

  class Box(Entity):
    pass

  class Place(Thing):
    pass

  class Position(Thing):
    pass

  class has_place(ObjectProperty, FunctionalProperty):
      domain = [Entity]
      range = [Place]

  class has_position(DataProperty, FunctionalProperty):
      domain = [Place]
      range = [str]

  class has_grabbed_box(ObjectProperty, FunctionalProperty):
      domain = [Robot]
      range = [Box]

  class has_boxes(DataProperty, FunctionalProperty):
      domain = [Container]
      range = [int]

  class has_capacity(DataProperty, FunctionalProperty):
      domain = [Container]
      range = [int]

  class has_target_box(DataProperty, FunctionalProperty):
      domain = [Robot]
      range = [Box]

  class is_claimed(ObjectProperty, FunctionalProperty):
      domain = [Box]
      range = [bool]

from logging.config import valid_ident
class RobotAgent(ap.Agent):

  """
    <-- Funcion de Inicializacion -->
  """
  def setup(self):
    self.agentType = 0
    self.pos = None
    self.target = None
    self.path = None
    self.has_box = False
    self.reserved_position = None
    self.first_step = True
    self.original_pos = None

    #Acciones
    self.actions = (
        self.pick_target,
        self.reserve_to_target,
        self.pick_up_box,
        self.pick_target_container,
        self.drop_box,
        self.return_to_base
    )

    #Reglas
    self.rules = (
        self.pick_target_rule,
        self.reserve_to_target_rule,
        self.pick_up_box_rule,
        self.pick_target_container_rule,
        self.drop_box_rule,
        self.return_to_base_rule
    )

  """
    <-- Funcion de Observacion -->
  """

  def see(self, e):
    self.pos = e.positions[self]
    if self.first_step:
      self.reserved_position = self.pos
      self.original_pos = self.pos
      self.first_step = False


  def next(self):
    for act in self.actions:
      for rule in self.rules:
        if rule(act):
          act()


  def step(self):
    #Ver el entorno
    self.see(self.model.grid)
    self.model.reservations.add_agents([self],[self.pos])
    self.next()

  def update(self):
    pass

  def end(self):
    pass

  """
    <-- Acciones -->
  """

  def pick_target(self):
    random_box = None

    min_distance = float('inf')
    for box in self.model.boxes:
      #print(self.model.grid.positions[box])
      distance = heuristic(self.pos,self.model.grid.positions[box])
      if distance <= min_distance and box.has_been_picked == False :
        min_distance = distance
        random_box = box

    if random_box == None:
      return
    random_box.has_been_picked = True
    self.target = self.model.grid.positions[random_box]
    #self.path = a_star(self.model.grid,self.pos,self.target)

  """
    Accion
  """

  def reserve_to_target(self):
      if self.reserved_position == self.target:
          return

      neighbors = get_neighbors(self.model.grid, self.pos)

      best_neighbor = None
      best_f_score = float('inf')

      for neighbor in neighbors:
          if neighbor in self.model.reservations.empty:
              tentative_g_score = heuristic(self.pos, neighbor) + 1
              neighbor_f_score = tentative_g_score + heuristic(neighbor, self.target)

              if neighbor_f_score < best_f_score:
                  best_f_score = neighbor_f_score
                  best_neighbor = neighbor

      if best_neighbor is not None:
          movement = np.subtract(best_neighbor, self.pos)
          self.model.grid.move_by(self, movement)
          self.model.reservations.remove_agents(self)
          self.reserved_position = best_neighbor
          self.model.reservations.add_agents([self], [self.reserved_position])


  def pick_up_box(self):
    self.has_box = True
    #print("BORRAMOS")
    #print(self.target)
    for agent in self.model.grid.agents:
      if self.model.grid.positions[agent] == self.pos and isinstance(agent, BoxAgent):
        self.model.grid.remove_agents(agent)
        self.model.boxes.remove(agent)
        break
    self.target = None

  def pick_target_container(self):
    """
    random_container = random.choice(model.containers)
    self.target = self.model.grid.positions[random_container]
    self.path = a_star(self.model.grid,self.pos,self.target)
    """

    random_container = None

    min_distance = float('inf')
    for container in self.model.containers :
      #print(self.model.grid.positions[container])
      distance = heuristic(self.pos,self.model.grid.positions[container])
      if distance <= min_distance and container.capacity != 0 : #ADD IS FULL HERE
        min_distance = distance
        random_container = container

    if random_container == None:
      return
    random_container.has_been_picked = True
    self.target = self.model.grid.positions[random_container]

  def drop_box(self):
    for agent in self.model.grid.agents:
      if self.model.grid.positions[agent] == self.pos and isinstance(agent, ContainerAgent):
        agent.capacity -= 1
        self.model.container_capacity -= 1
        break
    self.has_box = False
    self.target = None

  def return_to_base(self):
    self.target = self.original_pos

  """
    <-- Reglas -->
  """

  #Regla de objetivo
  def pick_target_rule(self,act):
    validator = [False,False,False,False]

    if self.target == None:
      validator[0] = True

    if self.has_box == False:
      validator[1] = True

    if len(self.model.boxes) != 0:
      validator[2] = True

    if act == self.pick_target:
      validator[3] = True

    return sum(validator) == 4

  #Regla de mover a objetivo
  def reserve_to_target_rule(self,act):
    valid_ident = [False,False, False]

    if self.target != None:
      valid_ident[0] = True

    if self.pos != self.target:
      valid_ident[1] = True

    if act == self.reserve_to_target:
      valid_ident[2] = True

    return sum(valid_ident) == 3

  def pick_up_box_rule(self,act):
    validator = [False,False,False]

    if self.pos == self.target:
      validator[0] = True

    if self.has_box == False:
      validator[1] = True

    if act == self.pick_up_box:
      validator[2] = True

    return sum(validator) == 3


  def pick_target_container_rule(self,act):
    validator = [False,False, False]

    if self.target == None:
      validator[0] = True

    if self.has_box == True:
      validator[1] = True

    if act == self.pick_target_container:
      validator[2] = True

    return sum(validator) == 3

  def drop_box_rule(self,act):
    validator = [False,False,False]

    if self.pos == self.target:
      validator[0] = True

    if self.has_box == True:
      validator[1] = True

    if act == self.drop_box:
      validator[2] = True

    return sum(validator) == 3

  def return_to_base_rule(self,act):
    validator = [False,False,False]

    if len(self.model.boxes) == 0:
      validator[0] = True

    if self.has_box == False:
      validator[1] = True

    if act == self.return_to_base:
      validator[2] = True

    return sum(validator) == 3

class BoxAgent(ap.Agent):

  """
    <-- Funcion de Inicializacion -->
  """
  def setup(self):
    self.agentType = 1
    self.has_been_picked = False
    self.pos = None

  def see(self, e):
    self.pos = e.positions[self]

  def next(self):
    pass

  def step(self):
    self.see(self.model.grid)

  def update(self):
    pass

  def end(self):
    pass  

class ContainerAgent(ap.Agent):
  """
    <-- Funcion de Inicializacion -->
  """
  def setup(self):
    self.agentType = 2
    self.capacity = 100

    pass

  def see(self, e):
    pass

  def next(self):
    pass

  def step(self):
    pass

  def update(self):
    pass

  def end(self):
    pass
  
class RobotModel(ap.Model):

  """
    <-- Funcion de Inicializacion -->
  """
  def setup(self):
    print("SUPA")
    self.steps = 0
    self.robots = ap.AgentList(self,len(self.p.robots),RobotAgent)
    self.boxes = ap.AgentList(self,len(self.p.boxes),BoxAgent)
    self.containers = ap.AgentList(self,len(self.p.containers),ContainerAgent)
    self.container_capacity = len(self.p.containers) * 5

    #Instancia Grid
    self.grid = ap.Grid(self, (self.p.M, self.p.N), track_empty=True)
    self.reservations = ap.Grid(self, (self.p.M, self.p.N), track_empty=True)

    #Asignacion de Agentes
    print(f"Robots: {self.p.robots}")
    print(f"Boxes: {self.p.boxes}")
    self.grid.add_agents(self.robots, positions=self.p.robots, empty=True)
    self.grid.add_agents(self.boxes, positions=self.p.boxes, empty=True)
    self.grid.add_agents(self.containers, positions=self.p.containers, empty=True)


  def step(self):
    print(f"Step: {self.steps}")
    self.robots.step()
    """
    for i in range(len(self.robots)):
      for j in range(i + 1, len(self.robots)):
        if self.grid.positions[self.robots[i]] == self.grid.positions[self.robots[j]]:
          print("Colision")
    """
    self.steps += 1
    print(f"Capacity: {self.container_capacity}")
    if self.container_capacity == 0:
      self.stop()

  def next(self):
    pass

  def update(self):
    self.record('Boxes Delivered', self.container_capacity)

  def end(self):
    pass

def animation_plot(model, ax):
    """
    Función de animación
    @param model: modelo
    @param ax: axes (matplotlib)
    """
    # Definición de atributo para tipo de agente
    agent_type_grid = model.grid.attr_grid('agentType')
    # Definición de gráfico con colores (de acuerdo al tipo de agente)
    ap.gridplot(agent_type_grid, cmap='Accent', ax=ax)
    # Definición de título del gráfico
    ax.set_title(f"Vaccum Model \n Time-step: {model.t}, "
                 f"Boxes: {0}")
    

#SIMULATION:

#Create figure (from matplotlib)
#fig, ax = plt.subplots()

#Create model
#model = RobotModel(parameters)
#model = None

#Run with animation
#If you want to run it without animation then use instead:
#model.run()
#animation = ap.animate(model, fig, ax, animation_plot)
#This step may take a while before you can see anything

#Print the final animation
#IPython.display.HTML(animation.to_jshtml())
def robotsModel(parameters):
    print(parameters['begin'] == 1)
    if parameters['begin'] == 1:
        print("vivo") 
        global model
        model = RobotModel(parameters)
        model.setup()
    else:
        print("sigo vivo")
    model.step()
    
    robots = []
    for robot in model.robots:
        posy = int(robot.pos[0])
        posx = int(robot.pos[1])
        robots.append({"id": robot.id, "position": [posy, posx]})
    
    containers = []
    for container in model.containers:
        containers.append({"id": container.id, "capacity": container.capacity})
    
    boxess = []
    for box in model.boxes:
        print(f"boxes: {box.pos}")
        print(f"idsfsdfsdfsd: {box.id}")
        print("aslkjdfalskdjf")
        boxess.append(box.pos)
    
    response = {"robots": robots, "containers": containers, "boxes": boxess}
    print(response)
    return response