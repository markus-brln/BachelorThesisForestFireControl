import math
import random as r
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

from pyastar import pyastar
from collections import deque
from Simulation.view.board import (
    Board,
)
from Simulation.performance import Performance

from .utility import (
    get_agent_location,
    get_fire_location,
    dirt,
    grass,
    house,
    water,
    layer,
    types,
)

from .constants import METADATA

WIDTH = METADATA['width']
HEIGHT = METADATA['height']
AGENTS = METADATA['agents']
'''
The entire environment consists of a WIDTH x HEIGHT matrix with a
depth layer for each important attribute of a cell:

- Grayscale color
- Temperature (how much heat it has gotten from burning neighbours)
- Heat (how much potential heat the cell can give off when burning)
- Fuel (for how many iterations it can burn)
- Threshold (the temperature it can maximally have before igniting)
- Fire mobility (whether the fire can spread over these cells)
- Agent position
- Cell type
'''


def create_map():
    gray = np.full((WIDTH, HEIGHT), grass['gray'])
    temp = np.zeros((WIDTH, HEIGHT))
    heat = np.full((WIDTH, HEIGHT), grass['heat'])
    fuel = np.full((WIDTH, HEIGHT), grass['fuel'])
    threshold = np.full((WIDTH, HEIGHT), grass['threshold'])
    f_mobility = np.ones((WIDTH, HEIGHT))
    a_mobility = np.ones((WIDTH, HEIGHT))
    agent_pos = np.zeros((WIDTH, HEIGHT))
    other_pos = np.zeros((WIDTH, HEIGHT))
    agent_hist_3 = np.zeros((WIDTH, HEIGHT))
    agent_hist_5 = np.zeros((WIDTH, HEIGHT))
    agent_hist_all = np.zeros((WIDTH, HEIGHT))
    agent_other_hist_3 = np.zeros((WIDTH, HEIGHT))
    agent_other_hist_5 = np.zeros((WIDTH, HEIGHT))
    agent_other_hist_all = np.zeros((WIDTH, HEIGHT))
    type_map = np.zeros((WIDTH, HEIGHT))
    return np.dstack((type_map, gray, temp, heat, fuel,
                      threshold, agent_pos, other_pos, agent_hist_3,
                      agent_hist_5, agent_hist_all, agent_other_hist_3,
                      agent_other_hist_5, agent_other_hist_all, f_mobility, a_mobility))


'''
Reset the layers of the map that have changed to default values
This function is also called after initialization

It can make a simple river from a starting position near the top
of the map randomly downwards to almost the bottom of the map, besides 
that some houses can be placed randomly on the map.
'''


def reset_map(env, make_river=True, make_houses=True):
    env[:, :, layer['gray']].fill(grass['gray'])
    env[:, :, layer['temp']].fill(0)
    env[:, :, layer['heat']].fill(grass['heat'])
    env[:, :, layer['fuel']].fill(grass['fuel'])
    env[:, :, layer['fire_mobility']].fill(1)
    env[:, :, layer['agent_mobility']].fill(1)
    env[:, :, layer['agent_pos']].fill(0)
    env[:, :, layer['other_pos']].fill(0)
    env[:, :, layer['agent_hist_3']].fill(0)
    env[:, :, layer['agent_hist_5']].fill(0)
    env[:, :, layer['agent_hist_all']].fill(0)
    env[:, :, layer['agent_other_hist_3']].fill(0)
    env[:, :, layer['agent_other_hist_5']].fill(0)
    env[:, :, layer['agent_other_hist_all']].fill(0)
    env[:, :, layer['type']].fill(0)

    if make_river:
        # Range of distances to keep from borders etc
        d = [1, 2, 3]
        # Remember where the fire is
        (fx, fy) = get_fire_location(WIDTH, HEIGHT)
        # Start anywhere on the x-axis
        river_x = np.random.choice(list(range(WIDTH)))
        # Start near the top on the y-axis but don't touch the border
        river_y = np.random.choice(d)
        # While we are not close to hitting the bottom of the y-axis
        while river_y < (HEIGHT - np.random.choice(d)):
            # Change the type
            env[river_x, river_y, layer['type']] = types['water']
            # Change the color
            env[river_x, river_y, layer['gray']] = water['gray']
            # Set the fire mobility to inf
            env[river_x, river_y, layer['fire_mobility']] = np.inf
            # Set the agent mobility to inf
            env[river_x, river_y, layer['agent_mobility']] = np.inf
            # Increment the position stochastically
            new_y = river_y + 1
            new_x = river_x + np.random.choice([1, 0, -1])
            while not np.random.choice(d) <= new_x < (WIDTH - np.random.choice(d)) \
                    and not (new_x, new_y) == (fx, fy):
                new_x = river_x + np.random.choice([1, 0, -1])
            (river_x, river_y) = (new_x, new_y)

    if make_houses:
        current_amount_of_houses = 0
        while current_amount_of_houses < METADATA['amount_of_houses']:
            # Get random X position to place house
            house_x = np.random.choice(list(range(WIDTH)))
            # Get random Y position to place house
            house_y = np.random.choice(list(range(HEIGHT)))
            # Change the type
            env[house_x, house_y, layer['type']] = types['house']
            # Change the color
            env[house_x, house_y, layer['gray']] = house['gray']
            # Set the fire mobility to 1
            env[house_x, house_y, layer['fire_mobility']] = 1
            # Set the agent mobility to inf
            env[house_x, house_y, layer['agent_mobility']] = np.inf
            # update amount of already placed houses
            current_amount_of_houses = current_amount_of_houses + 1


'''
 Agents can move, die and dig a cell to turn it into dirt
'''


class Agent:
    def __init__(self, W, position, active=False):
        self.W = W
        self.x, self.y = position
        self.dead = False
        self.digging = True
        self.FIRE_ISOLATED = False
        self.active = active
        self.agent_hist = deque()
        self.agent_hist.append((self.x, self.y))
        self.save_move = True

    '''
    Agent is dead if the cell at current position is burning
    *ATM This is switched off*
    '''

    def is_dead(self):
        if self.dead or self.W.is_burning((self.x, self.y)):
            if self.active:
                self.W.env[self.x, self.y, layer['agent_pos']] = 0
            else:
                self.W.env[self.x, self.y, layer['other_pos']] = 0
            return True
        return False

    '''
    Turn the cell at agent position into a dirt cell
    '''

    def dig(self):
        # Ignore dig command if we are not in dig mode
        if self.digging:
            # Ignore dig command if the cell is already a dirt cell
            if not types[self.W.env[self.x, self.y, layer['type']]] == 'dirt':
                # Change the type
                self.W.env[self.x, self.y, layer['type']] = types['dirt']
                # Change the color
                self.W.env[self.x, self.y, layer['gray']] = dirt['gray']
                # Set the fire mobility to inf
                self.W.env[self.x, self.y, layer['fire_mobility']] = np.inf

    '''
    Move the agent to a new position
    '''

    def move(self, direction):
        # Get the new position
        (nx, ny) = self._direction_to_coords(direction)
        # if new location is burning do nothing
        if not self.W.occupied(nx, ny):
            # Remove agent from old place
            self.W.env[self.x, self.y, layer['agent_pos']] = 0
            # If the new position is inbounds and traversable
            if self.W.inbounds(nx, ny) and self.W.traversable(nx, ny):
                if self.W.is_burning((nx, ny)):
                    self.save_move = False
                if not self.W.is_burning((nx, ny)):
                    # Go to new position
                    (self.x, self.y) = (nx, ny)
                    # Place agent on new position
                    self.W.env[self.x, self.y, layer['agent_pos']] = 1
                    self.agent_hist.append((self.x, self.y))
                    self.save_move = True

    '''
    Checks if the cell in the given direction is burning
    '''

    def fire_in_direction(self, direction):
        (nx, ny) = self._direction_to_coords(direction)
        return self.W.inbounds(nx, ny) and self.W.is_burning((nx, ny))

    '''
    Convert (N, S, E, W) or (0, 1, 2, 3) into coordinates
    '''

    def _direction_to_coords(self, direction):
        if direction in ["N", 0]:
            return self.x, self.y - 1
        if direction in ["S", 1]:
            return self.x, self.y + 1
        if direction in ["E", 2]:
            return self.x + 1, self.y
        if direction in ["W", 3]:
            return self.x - 1, self.y


'''
The world contains the map and everything else
'''


class World:
    def __init__(self, environment):
        # Set the environment (string)
        self.environment = environment
        # Create the map
        self.env = create_map()
        # Set dimensions
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        # Holds the locations of all agent (used to prevent 2 agent occupying the same position)
        self.locations = []
        # Set the rest of the params
        self.reset()
        # Set running and isolated
        self.RUNNING = True
        self.FIRE_ISOLATED = False
        # Create the agent(s) and let agent at pos 1 be active (first agent which will play
        self.agents = [
            Agent(self, get_agent_location(WIDTH, HEIGHT, self.locations)) for agent in range(AGENTS)
        ]

        # Clear locations (only used for initializing agents)
        self.locations.clear()
        # Set misc
        self.agents[0].active = True
        self.active = self.agents[0]
        self.board = None
        self.performance = Performance()

    '''
    Reset any changed parameters to their default values
    '''

    def reset(self):
        # Set wind parameters (not used in this project)
        if METADATA['wind'] == "random":
            self.wind_speed = np.random.choice([0, 0.7, 0.85])
            self.wind_vector = (r.randint(-1, 1), r.randint(-1, 1))
        else:
            self.wind_speed = METADATA['wind'][0]
            self.wind_vector = METADATA['wind'][1]

        # Simulation is running
        self.RUNNING = True
        self.FIRE_ISOLATED = False

        # Reset the map
        reset_map(self.env, METADATA['make_rivers'], METADATA['make_houses'])

        # Keep track of burning cells. A cell is represented by a tuple (x, y)
        self.burning_cells = set()
        self.set_fire_to(get_fire_location(WIDTH, HEIGHT))

        # Create the agent(s) and let agent at pos 1 be active (first agent which will play
        self.agents = [
            Agent(self, get_agent_location(WIDTH, HEIGHT, self.locations)) for agent in range(AGENTS)
        ]
        self.locations.clear()
        self.agents[0].active = True
        self.update_pov(True)


        # Keep track of all the points along the border (For the A* algorithm)
        self.reset_border_points()
        self.fire_at_border = False

        # set history for agents (3 previous states) will be the same at start
        for agent in self.agents:
            agent.agent_hist.append((agent.x, agent.y))

    def rebuild_initial_state(self, state):
        # Reset initial settings of the map
        self.env[:, :, layer['gray']].fill(grass['gray'])
        self.env[:, :, layer['temp']].fill(0)
        self.env[:, :, layer['heat']].fill(grass['heat'])
        self.env[:, :, layer['fuel']].fill(grass['fuel'])
        self.env[:, :, layer['fire_mobility']].fill(1)
        self.env[:, :, layer['agent_mobility']].fill(1)
        self.env[:, :, layer['agent_pos']].fill(0)
        self.env[:, :, layer['agent_hist']].fill(0)
        self.env[:, :, layer['type']].fill(0)

        # Simulation is running
        self.RUNNING = True
        self.FIRE_ISOLATED = False

        for y in range(state.shape[1]):
            for x in range(state.shape[1]):
                # Get initial agent pos
                if state[0, x, y, 0]:
                    # Create the agent(s)
                    self.agents = [
                        Agent(self, (x, y))
                    ]
                    self.env[x, y, layer['agent_pos']].fill(1)
                # Get initial water positions
                if state[0, x, y, 2]:
                    # Change the type
                    self.env[x, y, layer['type']] = types['water']
                    # Change the color
                    self.env[x, y, layer['gray']] = water['gray']
                    # Set the fire mobility to inf
                    self.env[x, y, layer['fire_mobility']] = np.inf
                    # Set the agent mobility to inf
                    self.env[x, y, layer['agent_mobility']] = np.inf

                # Get initial house positions
                if state[0, x, y, 3]:
                    # Change the type
                    self.env[x, y, layer['type']] = types['house']
                    # Change the color
                    self.env[x, y, layer['gray']] = house['gray']
                    # Set the fire mobility to 1
                    self.env[x, y, layer['fire_mobility']] = 1
                    # Set the agent mobility to inf
                    self.env[x, y, layer['agent_mobility']] = np.inf

        # Keep track of burning cells. A cell is represented by a tuple (x, y)
        self.burning_cells = set()
        self.set_fire_to(get_fire_location(WIDTH, HEIGHT))

        # Keep track of all the points along the border (For the A* algorithm)
        self.reset_border_points()
        self.fire_at_border = False

    '''
    Reset the queue containing the border points of the map
    '''

    def reset_border_points(self):
        self.border_points = deque()
        for x in range(WIDTH):
            self.border_points.append([x, 0])
            self.border_points.append([x, HEIGHT - 1])
        for y in range(HEIGHT):
            self.border_points.append([0, y])
            self.border_points.append([HEIGHT - 1, y])

    '''
    Determine whether a coordinate is within bounds of the map
    '''

    def inbounds(self, x, y):
        return x in range(self.WIDTH) and y in range(self.HEIGHT)

    '''
    Determine whether a coordinate is traversable by the agent (not water, house or burned house)
    '''

    def traversable(self, x, y):
        return self.env[x, y, layer['type']] != types['water'] and self.env[x, y, layer['type']] != types['house'] and \
               self.env[x, y, layer['gray']] != house['gray_house_burnt']

    '''
    Set a cell to be burning
    '''

    def set_fire_to(self, cell):
        x, y = cell
        # If statement prevents the situation in which the fire propagates while having digged just then
        if not self.env[x, y, layer['agent_pos']] and not self.env[x, y, layer['other_pos']]:
            # Set the temperature to be higher than the ignition threshold, if needed
            if self.env[x, y, layer['temp']] < self.env[x, y, layer['threshold']]:
                self.env[x, y, layer['temp']] = self.env[x, y, layer['threshold']] + 1
            # If fire set to house, set color to gray_house_burning
            if self.env[x, y, layer['type']] == types['house']:
                self.env[x, y, layer['gray']] = house['gray_house_burning']
            # Else the color is just gray_burning
            else:
                # Change the color
                self.env[x, y, layer['gray']] = grass['gray_burning']
            # Change the type
            self.env[x, y, layer['type']] = types['fire']
            # Update burning cells
            self.burning_cells.add(cell)
            # If this is a point at the border, the fire can never be contained
            if (x == 0) or (x == WIDTH - 1) or (y == 0) or (y == HEIGHT - 1):
                self.fire_at_border = True

    '''
    Returns whether the cell is on fire
    '''

    def is_burning(self, cell):
        x, y = cell
        return types[self.env[x, y, layer['type']]] == 'fire'

    '''
    Returns whether the cell can catch fire
    '''

    def is_burnable(self, cell):
        x, y = cell
        return types[self.env[x, y, layer['type']]] not in ['fire', 'burnt', 'dirt', 'water']
    '''
    Returns whether a cell blocks fire
    '''

    def is_blocking(self, cell):
        x, y = cell
        return types[self.env[x, y, layer['type']]] in ['burnt', 'dirt'] \
               or self.env[x, y, layer['agent_pos']] \
               or self.env[x, y, layer['other_pos']]

    '''
    Returns the distance and angle (relative to the wind direction when this functionality is used) between two cells
    '''

    def get_distance_and_angle(self, cell, other_cell, distance_metric):
        distance = None
        x, y = cell
        ox, oy = other_cell
        cx, cy = ox - x, oy - y  # Vector between cell and other_cell
        wx, wy = self.wind_vector

        # Manhattan distance: abs(x1 - x2) + abs(y1 - y2)
        if distance_metric == "manhattan":
            distance = abs(x - ox) + abs(y - oy)
        # Euclidean distance: sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
        elif distance_metric == "euclidean":
            distance = ((x - ox) ** 2 + (y - oy) ** 2) ** (0.5)
        # Angle between (wind_w, windy) and the vector between cell and other cell
        angle = abs(math.atan2(wx * cy - wy * cx, wx * cx + wy * cy))

        return distance, angle

    '''
    Applies heat from a burning cell to another cell (can ignite the other cell)
    '''

    def apply_heat_from_to(self, cell, other_cell):
        x, y = cell
        ox, oy = other_cell
        distance, angle = self.get_distance_and_angle(cell, other_cell, "manhattan")

        # Formula:
        # Heat = Cell_Heat * Wind_Speed * (Angle_to_wind + Distance_to_cell)^-1
        env_factor = (angle + distance) ** (-1)
        calculated_heat = self.wind_speed * self.env[x, y, layer['heat']] * env_factor

        # Heat up the cell with the calculated heat
        self.env[ox, oy, layer['temp']] += calculated_heat
        # If the cell heat exceeds the threshold, set fire to it
        if self.env[ox, oy, layer['temp']] > self.env[ox, oy, layer['threshold']]:
            self.set_fire_to(other_cell)

    '''
    Reduce the fuel of a cell, return true if successful and false if burnt out
    '''

    def reduce_fuel(self, cell):
        x, y = cell
        # Reduce fuel of burning cell
        self.env[x, y, layer['fuel']] -= 1
        # If burnt out, remove cell from burning cells and update color and type
        if self.env[x, y, layer['fuel']] <= 0:
            if self.env[x, y, layer['gray']] == house['gray_house_burning']:
                self.env[x, y, layer['gray']] = house['gray_house_burnt']
            else:
                self.env[x, y, layer['gray']] = grass['gray_burnt']
            self.env[x, y, layer['type']] = types['burnt']
            self.burning_cells.remove(cell)
            return False
        return True

    '''
    Returns every cell that can be reached by taking grass["radius"] manhattan
    distance steps from the origin and is burnable
    '''

    def get_neighbours(self, cell):
        cx, cy = cell
        neighbours = set()
        for x in range(grass['radius'] + 1):
            for y in range(grass['radius'] + 1 - x):
                if (x, y) == (0, 0):
                    continue
                # Get the cell in each quadrant
                cells = [(cx + x, cy + y),
                         (cx - x, cy + y),
                         (cx + x, cy - y),
                         (cx - x, cy - y)]
                for cell in cells:
                    if self.inbounds(*cell) and self.is_burnable(cell):
                        neighbours.add(cell)
        return neighbours

    '''
    Returned the score for a current state. Score isn't used to measure performance anymore. 
    Function analyzes whether the fire is isolated, therefore the function is kept.
    '''

    def get_reward(self):
        if self.outer_boarder_burning():
            self.RUNNING = False
        if len(self.burning_cells) == 0:
            self.FIRE_ISOLATED = True
        if not self.fire_at_border and len(self.border_points) \
                and len(self.burning_cells):

            # Make a copy of the burning cells set, and pop one out
            burning = set(self.burning_cells)
            b_cell = burning.pop()

            # Try finding a route from that burning cell to a border point
            grid = self.env[:, :, layer['fire_mobility']].astype(np.float32)
            start = np.array([b_cell[0], b_cell[1]])
            end = self.border_points.pop()
            path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

            # If that route is blocked, try other routes (other border points)
            while path.shape[0] == 0:
                # If there are no more routes from that burning cell to any border
                if len(self.border_points) == 0:
                    # If there are no more routes from any burning cell to any border
                    if len(burning) == 0:
                        # The fire is contained! Give a big reward
                        if METADATA['containment_wins']:
                            self.RUNNING = False
                            self.FIRE_ISOLATED = True
                        return 0
                    # No more border points: try a different burning cell
                    self.reset_border_points()
                    b_cell = burning.pop()
                    start = np.array([b_cell[0], b_cell[1]])
                # Still border points left, try a route to a different border point
                end = self.border_points.pop()
                path = pyastar.astar_path(grid, start, end, allow_diagonal=False)

            # Always put back the last working(!) end point for the route
            self.border_points.append(end)

        type_map = self.env[:, :, layer['type']]
        # Agent died
        if not self.agents:
            num_healthy_cells = np.count_nonzero(type_map == types['water'])
            # only healthy cells are the ones that can never burn
            percent_healthy = num_healthy_cells / (WIDTH * HEIGHT)
            return 1000 * percent_healthy
        else:
            # Agent survived
            num_destroyed_cells = np.count_nonzero(type_map == types['dirt']) + \
                                  np.count_nonzero(self.env[:, :, layer['gray']] == house['gray_house_burnt']) + \
                                  np.count_nonzero(self.env[:, :, layer['gray']] == grass['gray_burnt'])

            percent_healthy = 1 - (num_destroyed_cells / (WIDTH * HEIGHT))
            return 1000 * percent_healthy

    '''
    Return the state of the simulation:
    State consists of combinations of different layers with each containing only boolean entries
    - all agent positions an histories
    - The fire positions
    - The dirt positions
    '''

    def get_state(self):
        if self.environment == 'forest':
            return np.dstack((self.env[:, :, layer['agent_pos']],
                              self.env[:, :, layer['other_pos']],
                              self.env[:, :, layer['agent_hist_3']],
                              self.env[:, :, layer['agent_hist_5']],
                              self.env[:, :, layer['agent_hist_all']],
                              self.env[:, :, layer['agent_other_hist_3']],
                              self.env[:, :, layer['agent_other_hist_5']],
                              self.env[:, :, layer['agent_other_hist_all']],
                              self.env[:, :, layer['type']] == types['fire'],
                              self.env[:, :, layer['type']] == types['dirt'],
                              self.env[:, :, layer['type']] == types['grass'],
                              ))

    '''
    Returns a 90,180 and 270 roation of the state. Augmented data can then
    be remembered
    '''

    def get_augmented_state(self):
        rot90_1 = np.dstack((np.rot90(self.env[:, :, layer['agent_pos']]),
                             np.rot90(self.env[:, :, layer['other_pos']]),
                             np.rot90(self.env[:, :, layer['agent_hist_3']]),
                             np.rot90(self.env[:, :, layer['agent_hist_5']]),
                             np.rot90(self.env[:, :, layer['agent_hist_all']]),
                             np.rot90(self.env[:, :, layer['agent_other_hist_3']]),
                             np.rot90(self.env[:, :, layer['agent_other_hist_5']]),
                             np.rot90(self.env[:, :, layer['agent_other_hist_all']]),
                             np.rot90(self.env[:, :, layer['type']] == types['fire']),
                             np.rot90(self.env[:, :, layer['type']] == types['dirt']),
                             np.rot90(self.env[:, :, layer['type']] == types['grass']),
                             ))

        rot90_2 = np.dstack((np.rot90(self.env[:, :, layer['agent_pos']], k=2),
                             np.rot90(self.env[:, :, layer['other_pos']], k=2),
                             np.rot90(self.env[:, :, layer['agent_hist_3']], k=2),
                             np.rot90(self.env[:, :, layer['agent_hist_5']], k=2),
                             np.rot90(self.env[:, :, layer['agent_hist_all']], k=2),
                             np.rot90(self.env[:, :, layer['agent_other_hist_3']], k=2),
                             np.rot90(self.env[:, :, layer['agent_other_hist_5']], k=2),
                             np.rot90(self.env[:, :, layer['agent_other_hist_all']], k=2),
                             np.rot90(self.env[:, :, layer['type']] == types['fire'], k=2),
                             np.rot90(self.env[:, :, layer['type']] == types['dirt'], k=2),
                             np.rot90(self.env[:, :, layer['type']] == types['grass'], k=2),

                             ))

        rot90_3 = np.dstack((np.rot90(self.env[:, :, layer['agent_pos']], k=3),
                             np.rot90(self.env[:, :, layer['other_pos']], k=3),
                             np.rot90(self.env[:, :, layer['agent_hist_3']], k=3),
                             np.rot90(self.env[:, :, layer['agent_hist_5']], k=3),
                             np.rot90(self.env[:, :, layer['agent_hist_all']], k=3),
                             np.rot90(self.env[:, :, layer['agent_other_hist_3']], k=3),
                             np.rot90(self.env[:, :, layer['agent_other_hist_5']], k=3),
                             np.rot90(self.env[:, :, layer['agent_other_hist_all']], k=3),    
                             np.rot90(self.env[:, :, layer['type']] == types['fire'], k=3),
                             np.rot90(self.env[:, :, layer['type']] == types['dirt'], k=3),
                             np.rot90(self.env[:, :, layer['type']] == types['grass'], k=3),

                             ))
        if self.environment == 'forest':
            return rot90_1, rot90_2, rot90_3

    '''
    Returns a list of all non active agents
    '''

    def other_agents(self):
        other = []
        for agent in self.agents:
            other.append(agent)
        return other

    '''
    Returns True if position is occupied, False otherwise
    '''

    def occupied(self, x, y):
        for agent in self.agents:
            if (agent.x, agent.y) == (x, y):
                return True
        return False

    '''
    Updates the point of view of the system. We want the action of the agent to
    be matched to the right state. so the layer "agent_pos" represents the agent 
    which most recently made a move and other_pos is represents the other agents.
    '''

    def update_pov(self, reset):
        self.fill_hist_with_0()
        # if update pov is called from "reset" function update for pov of the active agent
        if reset:
            for agent in self.agents:
                if agent.active:
                    self.env[agent.x, agent.y, layer['agent_pos']] = 1
                    self.update_agent_hist(agent)
                else:
                    self.env[agent.x, agent.y, layer['other_pos']] = 1
                    self.update_other_agent_hist(agent)
        # else we update it for the next active agent
        else:
            if len(self.agents) == 1:
                self.env[self.agents[0].x, self.agents[0].y, layer['agent_pos']] = 1
                self.update_agent_hist(self.agents[0])
            else:
                for agent in self.agents:
                    if agent == self.next_agent():
                        self.env[agent.x, agent.y, layer['agent_pos']] = 1
                        self.update_agent_hist(agent)
                    else:
                        self.env[agent.x, agent.y, layer['other_pos']] = 1
                        self.update_other_agent_hist(agent)

    '''
    Updates the layers containing the (active) agent history
    '''

    def update_agent_hist(self, agent):
        cnt = 0
        for (x, y) in reversed(agent.agent_hist):
            if cnt <= 3:
                self.env[x, y, layer['agent_hist_3']] = 1
            if cnt <= 5:
                self.env[x, y, layer['agent_hist_5']] = 1
            self.env[x, y, layer['agent_hist_all']] = 1
            cnt += 1

    '''
    update the layer containing the (other) agent history
    '''

    def update_other_agent_hist(self, agent):
        cnt = 0
        for (x, y) in reversed(agent.agent_hist):
            if cnt <= 3:
                self.env[x, y, layer['agent_other_hist_3']] = 1
            if cnt <= 5:
                self.env[x, y, layer['agent_other_hist_5']] = 1
            self.env[x, y, layer['agent_other_hist_all']] = 1
            cnt += 1

    '''
    Fills all agent layers will zeros
    '''

    def fill_hist_with_0(self):
        self.env[:, :, layer['agent_pos']].fill(0)
        self.env[:, :, layer['other_pos']].fill(0)
        self.env[:, :, layer['agent_hist_3']].fill(0)
        self.env[:, :, layer['agent_hist_5']].fill(0)
        self.env[:, :, layer['agent_hist_all']].fill(0)
        self.env[:, :, layer['agent_other_hist_3']].fill(0)
        self.env[:, :, layer['agent_other_hist_5']].fill(0)
        self.env[:, :, layer['agent_other_hist_all']].fill(0)

    '''
    Returns next active agent
    '''

    def next_agent(self):

        if self.agents.index(self.active) + 2 > len(self.agents):
            return self.agents[0]
        else:
            return self.agents[self.agents.index(self.active) + 1]

    '''
    used to initialize video system for pygame
    '''

    def set_board(self, view):
        if view == "True":
            self.board = Board(self)

    '''
    double check if the fire is contained from all sides i.e either dirt or burnt or one of the agents on all sides
    '''

    def double_check_if_isolated(self):
        if len(self.burning_cells) == 1:
            for i in self.burning_cells:
                n = self.get_neighbours(i)
                for item in n:
                    if not self.is_blocking(item):
                        return False
            return True

    '''
    returns if if the fire has reached the outer border i.e. the end of the map 
    (fire out of control)
    '''

    def outer_boarder_burning(self):
        for item in self.get_outer_boarder_points():
            x, y = item
            if self.env[x, y, layer['type']] == types['fire'] == 1:
                return True

    '''
    returns  the coordinates of all outermost boarder cells
    '''

    def get_outer_boarder_points(self):
        points = []
        for cor in range(self.WIDTH):
            points.append((0, cor))
            points.append((self.HEIGHT - 1, cor))
            points.append((cor, 0))
            points.append((cor, self.WIDTH - 1))
        return points

    def get_percent_burnt(self):
        a = self.env[:, :, layer['type']] == types['burnt']
        burnt = np.sum(a == 1)
        return (burnt/(self.WIDTH * self.WIDTH)) * 10