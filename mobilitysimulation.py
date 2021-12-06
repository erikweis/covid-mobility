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
import pandas as pd

from parameter_sweep import Simulation, Experiment

from agent import Agent
from job import Job
from location import Location
from tools.utils import get_object_params
# from tools.initialize_locations import initialize_grid
# from tools.initialize_agents import place_grid_agents

from initial_state import get_initial_capacities

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
        proportion_remote_workers: float = 1.0,
        covid_intervention_time = None,
        **kwargs
    ): 

        """[summary]
        """

        self._dirname = dirname

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_steps = num_steps
        self.proportion_remote_workers = proportion_remote_workers
        self.covid_intervention_time = covid_intervention_time
        
        #initialize locations
        total_occupancy = 0.99
        total_capacity = num_agents/total_occupancy
        num_cities = int(grid_size**2/10)
        capacities = get_initial_capacities(grid_size,num_cities,total_capacity)

        N = grid_size
        grid = np.empty((N,N),dtype=Location)
        idx = 0
        for i in range(N):
            for j in range(N):
                grid[i][j] = Location(idx,coords=(i,j),capacity = int(capacities[i][j]))
                idx += 1

        self.locations = grid

        #initialize agents
        self.agents = []
        for agentID in range(self.num_agents):
            
            loc = np.random.choice(grid.flatten()) #,p=capacities.flatten()/sum(capacities.flatten()))
            job = Job(idx = agentID,salary=np.random.exponential(scale=30000))
            a = Agent(agentID,loc,job)
            self.agents.append(a)

        #data
        self.agent_location_data = []
        self.move_data = []
        self.location_score_data = []
        self.move_decision_score_data = []


    @property
    def dirname(self):
        return self._dirname


    def move_agent(self,agent,time):

        old_loc, new_loc, possible_location_scores = agent.decide_where_to_move(self.locations)
        agent.location = new_loc #update agent location
        old_loc.remove_agent(agent) #remove agent from old location
        new_loc.add_agent(agent) #add agent to new location
        
        #save move data
        self.move_data.append([time,agent.idx,old_loc.idx,*old_loc.coords,new_loc.idx,*new_loc.coords])

        # save location score data
        move_dict = {'time':time,'fromlocID':old_loc.idx,'tolocID':new_loc.idx}
        for loc_score_dict in possible_location_scores:
            combo_dict = {**move_dict,**loc_score_dict}
            self.location_score_data.append(combo_dict)

    def update(self,t):

        # add remote work with covid at a particular time step, if specified
        if t == self.covid_intervention_time:
            for agent in self.agents:
                if random.random() < self.proportion_remote_workers:
                    agent.job.remote_status = True
        
        #loop over agents
        for agent in self.agents:
            decision, score_dict = agent.decide_to_move()
            if decision:
                self.move_agent(agent,time=t)

            self.agent_location_data.append([t,*agent.get_tidy_state()]) #save state to data
            self.move_decision_score_data.append({'time':t,'agent_id':agent.idx,**score_dict})

    def run_simulation(self):

        for t in tqdm(range(self.num_steps)):
            self.update(t)


    def save_state(self):

        for agent in self.agents:
            self.agent_location_data.append(agent.get_tidy_state())


    def save_data(self):
        
        #locations
        fpath_locs = os.path.join(self.dirname,'locations.jsonl')
        location_params_to_ignore = ['agents','quality']
        loc_params = [ get_object_params(loc,location_params_to_ignore) for loc in self.locations.flatten()]
        lines = [json.dumps(lp) for lp in loc_params]
        with open(fpath_locs,'w') as f:
            f.write('\n'.join(lines))
        
        #agents
        fpath_agents = os.path.join(self.dirname,'agents.jsonl')
        agent_params_to_ignore = ['location_data','job']
        agent_params = [get_object_params(agent,agent_params_to_ignore) for agent in self.agents]
        for agent_dict,agent in zip(agent_params,self.agents):
            agent_dict['income'] = int(agent.income)
        lines = [json.dumps(ap) for ap in agent_params]
        with open(fpath_agents,'w') as f:
            f.write('\n'.join(lines))

        #global params
        fpath_params = os.path.join(self.dirname,'params.json')
        global_params_to_ignore = ['locations','agents','data','location_score_data','move_decision_score_data']
        global_params = get_object_params(self,global_params_to_ignore)
        with open(fpath_params,'w') as f:
            json.dump(global_params,f)

        #save data
        fpath_data = os.path.join(self.dirname,'agent_location_data.csv')
        with open(fpath_data,'w') as csvfile:
            writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['time','agentID','locationID','xloc','yloc'])
            for row in self.agent_location_data:
                writer.writerow(row)

        fpath_move_data = os.path.join(self.dirname,'move_data.csv')
        with open(fpath_move_data,'w') as csvfile:
            writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['time','agentID','fromlocID','from_x','from_y','tolocID','to_x','to_y'])
            for row in self.move_data:
                writer.writerow(row)

        fpath_location_score_data = os.path.join(self.dirname,'location_score_data.csv')
        df = pd.DataFrame(self.location_score_data)
        df.to_csv(fpath_location_score_data)

        fpath_move_decision_score_data = os.path.join(self.dirname,'move_decision_score_data.csv')
        df = pd.DataFrame(self.move_decision_score_data)
        agg = {
            'total_score':'mean',
            'score_income': 'mean',
            'score_income_match': 'mean',
            'score_housing_cost': 'mean'
        }
        df = df.groupby('time').agg(agg)
        df.to_csv(fpath_move_decision_score_data)


    def on_finish(self):
        self.save_data()


if __name__ == "__main__":

    random.seed(1)

    e = Experiment(
        MobilitySimulation,
        'movement1',
        root_dir = 'data',
        grid_size=[20],
        num_steps = [200],
        num_agents = [10000],
        covid_intervention_time = [100]
    )
    e.run_all_trials(debug=True)

