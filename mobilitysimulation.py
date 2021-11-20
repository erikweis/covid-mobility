import random
import numpy as np
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


class MobilitySimulation(Simulation):

    def __init__(
        self,
        dirname,
        num_agents:int = 100,
        grid_size:int = 20,
        num_steps:int = 500,
        **kwargs
    ): 

        self._dirname = dirname

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_steps = num_steps
        
        #initialize locations
        self.locations = np.array([Location(idx, capacity =10) for idx in \
                            range(grid_size**2)]).reshape((grid_size,grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                self.locations[i][j].coords = (i,j)
                
        #initialize agents
        self.agents = [Agent(idx=i,location=random.choice(self.locations.flatten())) for i in range(self.num_agents)]

        #data
        self.data = []


    @property
    def dirname(self):
        return self._dirname


    def update(self):

        for agent in self.agents:
            if agent.decide_to_move():
                old_loc, new_loc = agent.decide_where_to_move(self.locations)
                agent.location = new_loc #update agent location
                old_loc.remove_agent(agent) #remove agent from old location
                new_loc.add_agent(agent) #add agent to new location

            self.data.append(agent.get_tidy_state())


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
        agent_params_to_ignore = ['location_data']
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
        'random_movement_experiment',
        grid_size=[10,20,30]
    )
    e.run_all_trials(debug=True)
