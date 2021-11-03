
import random
import numpy as np
from tqdm import tqdm
import os
import datetime
import csv
import json
import networkx as nx
import copy

from person import Person
from idea import Idea

class Ecosystem:

    def __init__(self,
        num_people=30,
        num_seed_ideas=10,
        belief_strength_max = 10,
        k_nearest = 5,
        p_rewire = 2,
        threshold_agree=0.9,
        threshold_intrigued=0.6,
        num_initial_exposed_ideas=5):
        
        #ideas
        ideas = [Idea(idx=i,length=12) for i in range(num_seed_ideas)]
        self._num_ideas = num_seed_ideas
        self._idx2ideas = {idea.idx: idea for idea in ideas}
        self.belief_strength_max = belief_strength_max

        #create social network
        k_nearest = 5
        p_rewire = 0.2
        graph_params = {"n":num_people,"k":k_nearest,"p":p_rewire}
        small_world = nx.watts_strogatz_graph(num_people,k_nearest,p_rewire,seed=1)
        weighted_edges = [(i,j,1) for i,j in small_world.edges]
        weighted_edges += [(j,i,1) for i,j in small_world.edges]
        self.D_sn = nx.DiGraph()
        self.D_sn.add_weighted_edges_from(weighted_edges)

        self.people = []

        for idx in range(num_people):
            exposed_ideas = random.sample(list(self._idx2ideas.keys()),num_initial_exposed_ideas)
            p = Person(
                idx = idx,
                exposed_ideas = exposed_ideas,
                threshold_agree=threshold_agree,
                threshold_intrigued=threshold_intrigued,
                ecosystem=self,
                belief_strength_max=belief_strength_max,
                creativity=0.5)
            self.people.append(p)

        self.idx2people = {p.idx:p for p in self.people}
        self.time = 0

        self.beliefs_data = []
        self.idea_popularity_data = []
        self.social_networks = []

        self.run_params = {
            'num_seed_ideas':num_seed_ideas,
            'num_people':num_people,
            'graph_params':graph_params,
        }

        self.save_state()

    def num_ideas(self):
        return self._num_ideas

    def get_idx2ideas(self):
        return self._idx2ideas      

    def _evolve(self):

        for person in self.people:
            person.update(self.idx2people,self.D_sn)
        self.time += 1
        self.save_state()

    def evolve(self,steps=1):

        for _ in tqdm(range(steps)):
            self._evolve()

    def all_ideas(self):
        return self._idx2ideas.values()

    def all_idea_idxs(self):
        return self._idx2ideas.keys()

    def add_idea(self,idea_bitstring):
        
        new_idea_idx = self._num_ideas
        new_idea = Idea(idx=new_idea_idx,bitstring=idea_bitstring)
        self._idx2ideas[new_idea_idx] = new_idea
        self._num_ideas += 1

        return new_idea_idx

    def save_state(self):
        
        ideaidx2popularity = {idx:0 for idx in self.get_idx2ideas().keys()} 

        for person in self.people:
            for belief_idx, belief_strength in person.beliefs.items():
                
                #add row to data
                row = [self.time,person.idx,belief_idx,belief_strength]
                self.beliefs_data.append(row)
                
                #contribute to popularity score
                ideaidx2popularity[belief_idx] += belief_strength

        #save network state
        self.social_networks.append(copy.deepcopy(self.D_sn))
        
        for ideaID,popularity in ideaidx2popularity.items():
            self.idea_popularity_data.append([self.time,ideaID,popularity])

    def collect_run_params(self):

        data = self.run_params
        data['simulation_length']=self.time
        return data

    def save_data(self):

        dirname = 'data/'+datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        os.mkdir(dirname)

        idea_pop_fp = dirname+'/idea_popularity.csv'
        beliefs_fp = dirname+'/beliefs.csv'
        run_params = dirname+'/run_params.json'
        ideas_fp = dirname+'/ideas.csv'
        
        #network directory
        networks_dir = os.path.join(dirname,'networks')
        os.mkdir(networks_dir)
        
        for t,D in enumerate(self.social_networks):
            path = os.path.join(networks_dir,f't{t}.edgelist')
            nx.write_weighted_edgelist(D,path) 

        with open(ideas_fp,'w',newline='') as csvfile:
            writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['ideaID','bitstring'])
            for ideaID,idea in self.get_idx2ideas().items():
                writer.writerow([ideaID,idea.bitstring.to01()])

        with open(beliefs_fp, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['time','personID','beliefID','belief_strength'])
            for row in self.beliefs_data:
                writer.writerow(row)

        with open(idea_pop_fp, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile,quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['time','ideaID','popularity'])
            for row in self.idea_popularity_data:
                writer.writerow(row)

        with open(run_params,'w') as f:
            json.dump(self.collect_run_params(),f)

        with open(dirname+ '/person_params.jsonl','w') as f:
            for person in self.people:
                json.dump(person.get_all_params(),f)
                f.write('\n')

        print("Finished writing data to data files.")


if __name__=="__main__":

    #random.seed(10)
    random.seed(1)

    e = Ecosystem(
        num_people=30,
        num_seed_ideas=10,
        belief_strength_max = 10,
        k_nearest = 5,
        p_rewire = 2,
        threshold_agree=0.85,
        threshold_intrigued=0.4,)
    e.evolve(300)
    e.save_data()
    