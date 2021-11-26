import random
import numpy as np
from numpy.lib.function_base import place
from tqdm import tqdm
import os
import datetime
import csv
import json
import networkx as nx
import copy
from parameter_sweep import Simulation, Experiment

from agent import Agent
from job import Job
from location import Location
from tools.utils import get_object_params
from tools.initialize_locations import initialize_grid
from tools.initialize_agents import place_grid_agents

class MobilitySimulation(Simulation):

    """MobilitySimulation is the central class from running a mobility simulation.

    To run, initialize the object, then call the method `run_simulation()`. To save the data,
    call the `save_data()` method.

    This class extents the `Simulation` object from abm-parameter-sweep, such that it can
    be used with that package to run multiple simulations automatically.
    """

    def __init__(
        self,
        dirname,
        num_agents:int = 100,
        grid_size:int = 20,
        num_steps:int = 500,
        **kwargs
    ): 

        """[summary]
        """

        self._dirname = dirname

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_steps = num_steps
        
        #initialize locations
        # self.locations = np.array([Location(idx, capacity =10) for idx in \
        #                     range(grid_size**2)]).reshape((grid_size,grid_size))
        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         self.locations[i][j].coords = (i,j)

        #initialize grid and agents
        grid, mean, std_dev, capacities = initialize_grid(size=grid_size)
        grid, agents = place_grid_agents(grid)
        self.locations = grid
        self.agents = agents

        #data
        self.data = []


    @property
    def dirname(self):
        return self._dirname


    def move_agent(self,agent):

        old_loc, new_loc = agent.decide_where_to_move(self.locations)
        agent.location = new_loc #update agent location
        old_loc.remove_agent(agent) #remove agent from old location
        new_loc.add_agent(agent) #add agent to new location

    def update(self):

        for agent in self.agents:
            if agent.decide_to_move():
                self.move_agent(agent)

            self.data.append(agent.get_tidy_state()) #save state to data


    def run_simulation(self):

        for i in tqdm(range(self.num_steps)):
            self.update()


    def save_state(self):

        for agent in self.agents:
            self.data.append(agent.get_tidy_state())


    def save_data(self):
        
        #locations
        fpath_locs = os.path.join(self.dirname,'locations.jsonl')
        location_params_to_ignore = ['agents']
        loc_params = [ get_object_params(loc,location_params_to_ignore) for loc in self.locations.flatten()]
        lines = [json.dumps(lp) for lp in loc_params]
        with open(fpath_locs,'w') as f:
            f.write('\n'.join(lines))
        
        #agents
        fpath_agents = os.path.join(self.dirname,'agents.jsonl')
        agent_params_to_ignore = ['location_data','job']
        agent_params = [get_object_params(agent,agent_params_to_ignore) for agent in self.agents]
        lines = [json.dumps(ap) for ap in agent_params]
        with open(fpath_agents,'w') as f:
            f.write('\n'.join(lines))

        #global params
        fpath_params = os.path.join(self.dirname,'params.json')
        global_params_to_ignore = ['locations','agents','data']
        global_params = get_object_params(self,global_params_to_ignore)
        with open(fpath_params,'w') as f:
            json.dump(global_params,f)

        #save data
        fpath_data = os.path.join(self.dirname,'data.csv')
        with open(fpath_data,'w') as csvfile:
            writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['time','agentID','xloc','yloc'])
            for row in self.data:
                writer.writerow(row)


    def on_finish(self):
        self.save_data()


if __name__ == "__main__":

    random.seed(1)

    e = Experiment(
        MobilitySimulation,
        'random_movement_with_denis_initialization',
        root_dir = 'data',
        grid_size=[20,40],
        num_steps = [100]
    )
    e.run_all_trials(debug=True)

