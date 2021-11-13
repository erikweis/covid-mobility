import random
import numpy as np
from tqdm import tqdm
import os
import datetime
import csv
import json
import networkx as nx
import copy

from agent import Agent
from job import Job
from location import Location

class Ecosystem:

    def __init__(
        self,
        num_agents:int = 100,
        grid_size:int = 20,
        num_steps:int = 500,
        **kwargs
    ): 

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_steps = num_steps
        
        capacities = self.initialize_capacities()
        self.locations = [Location(i,j,capacity=capacities[i][j]) for i in range(grid_size) for j in range(grid_size)]
        self.agents = [Agent(idx=i,location=random.choice(self.locations)) for i in range(self.num_agents)]


    def initialize_capacities(self):
        
        #replace with smoothing function
        return np.random.randint(5,50,size=(self.grid_size,self.grid_size))


    def update(self):

        for agent in self.agents:
            if agent.decide_to_move():
                agent.move(self.locations)

    def run_simulation(self):

        for i in tqdm(range(self.num_steps)):
            self.update()

    def save_data():
        pass


if __name__ == "__main__":

    e = Ecosystem

