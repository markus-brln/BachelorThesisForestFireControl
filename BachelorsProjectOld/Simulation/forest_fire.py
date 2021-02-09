from termcolor import colored

from Simulation.constants import (
    METADATA,
)

from Simulation.utility import (
    layer,
    get_name,
    color2ascii,
    types,
)

from Simulation.environment import (
    World,
)

from Simulation.view.view import (
    draw,
)


def at_pos(other_agents, x, y):
    for agent in other_agents:
        if (agent.x, agent.y) == (x, y):
            return True
    return False


class ForestFire:
    def __init__(self, environment):
        self.W = World(environment)
        self.METADATA = METADATA
        self.DEBUG = METADATA['debug']
        self.layer = layer
        self.get_name = get_name
        self.width = METADATA['width']
        self.height = METADATA['height']
        self.n_actions = METADATA['n_actions']

    '''
    Execute an action in the environment
    '''
    def step(self, action):
        active, other = self.active_other_agents()

        # Handle basic movement actions
        if action in ["N", "S", "E", "W"] or action in range(4):
            self.W.agents[self.W.agents.index(active)].move(action)
        # Handle the dig action
        if METADATA['allow_dig_toggle'] and action in ["D", 4]:
            active.dig()
        # Update environment only every AGENT_SPEED steps
        METADATA['a_speed_iter'] -= 1
        if METADATA['a_speed_iter'] == 0:
            # if self.W.agents[len(self.W.agents) - 1].active:
            self.handle_fps()
            METADATA['a_speed_iter'] = METADATA['a_speed']
        self.W.update_pov(False)
        # Return the state, reward and whether the simulation is still running
        return [self.W.get_state(),
                self.W.get_reward(),
                not self.W.RUNNING,
                {}]

    '''
    Reset
    '''

    def reset(self):
        self.W.reset()
        return self.W.get_state()

    # Reset the simulation to its initial state
    def initial_state(self, initial_state):
        self.W.rebuild_initial_state(initial_state)
        return self.W.get_state()

    '''
    Print an ascii rendering of the simulation
    Prinsts agnet in color. Red = Active agent.
    If agent is on top of a digged whole the agent = Green. 
    '''

    def render(self, view='False', draw_terminal=True):
        return_map = "\n"
        if view == 'True':
            draw(self.W.board, self.W)
        elif view == "False" and draw_terminal:
            active, other = self.active_other_agents()
            # Print index markers along the top
            print(" ", end="")
            for x in range(self.width):
                print(x % 10, end="")
            print("")

            return_map = "\n"
            for y in range(self.height):
                # Print index markers along the left side
                print(y % 10, end="")
                for x in range(self.width):
                    # If the agent is at this location, print A
                    if self.W.agents and (active.x, active.y) == (x, y):
                        if self.W.env[x, y, layer['type']] == types['dirt']:
                            print(colored('A', 'green'), end="")
                        else:
                            print(colored('A', 'red'), end="")
                            return_map += 'A'
                    elif self.W.agents and at_pos(other, x, y):
                        if self.W.env[x, y, layer['type']] == types['dirt']:
                            print(colored('a', 'green'), end="")
                        else:
                            print('a', end="")
                            return_map += 'a'
                    # Otherwise use the ascii mapping to print the correct symbol
                    else:
                        symbol = color2ascii[self.W.env[x, y, layer['gray']]]
                        return_map += symbol
                        if x == round(self.width/2) or y == round(self.width/2):
                            print(colored(symbol, 'blue'), end="")
                        # if x == 10 and y == 20 or x == 20 and y == 10 or x == 30 and y == 20 or x == 20 and y == 30 :
                        #     print(colored(symbol, 'yellow'), end="")
                        else:
                            print(symbol, end="")
                return_map += '\n'  
                print("")
            print("")

        # Return a string representation of the map incase we want to save it
        return return_map

    '''
    Updates the simulations internal state
    '''
    def update(self):
        # Iterate over a copy of the set, to avoid ConcurrentModificationException
        burning = list(self.W.burning_cells)
        # For each burning cell
        for cell in burning:
            # Reduce it's fuel. If it has not burnt out, continue
            # Burnt out cells are removed automatically by this function
            if self.W.reduce_fuel(cell):
                # For each neighbour of the (still) burning cell
                for n_cell in self.W.get_neighbours(cell):
                    # If that neighbour is burnable
                    if self.W.is_burnable(n_cell):
                        # Apply heat to it from the burning cell
                        # This function adds the n_cell to burning cells if it ignited
                        self.W.apply_heat_from_to(cell, n_cell)

        # Simulation is terminated when there are no more burning cells or fire isolated
        if not self.W.burning_cells or self.W.FIRE_ISOLATED:
            self.W.RUNNING = False

    def handle_fps(self):
        if len(self.W.agents) == 1:
            if METADATA['fire_propagation_speed'] == 1:
                self.update()
            if METADATA['fire_propagation_speed'] == 2:
                self.update()
                self.update()
        else:
            if METADATA['fire_propagation_speed'] == 2:
                update = round(len(self.W.agents)/2)
                if self.W.agents[len(self.W.agents) - 1].active or self.W.agents[update - 1].active:
                    self.update()
            if METADATA['fire_propagation_speed'] == 1:
                if self.W.agents[len(self.W.agents) - 1].active:
                    self.update()


    '''
    returns the active agent and a list of the other agents
    '''

    def active_other_agents(self):
        other = []
        active = None
        for agent in self.W.agents:
            if agent.active:
                active = agent
            else:
                other.append(agent)
        if active is None:
            active = self.W.agents[0]
        return active, other

    '''
    returns whether agents is dead or not
    '''

    def agent_dead(self):
        for agent in self.W.agents:
            if agent.dead:
                return True
        return False
