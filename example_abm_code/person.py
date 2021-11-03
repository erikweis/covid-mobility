from re import S
from typing import Iterable
from idea import Idea
import random
import numpy as np
from bitarray import bitarray
import secrets
import scipy.stats

class Person:

    """
    Person class

    Args:
        beliefs (dict): dictionary of idea_id : belief strength
        
    """

    def __init__(self,
        idx:int,
        beliefs: dict = None,
        exposed_ideas = None,
        max_belief_capacity: int = 100,
        ecosystem = None,
        creativity = 0.1,
        threshold_agree=0.9,
        threshold_intrigued=0.6,
        belief_strength_max = 10):

        assert all([isinstance(x,int) for x in exposed_ideas])

        self.idx = idx
        self._beliefs = {idea_idx:1 for idea_idx in exposed_ideas}
        self.ecosystem = ecosystem
        self.creativity = creativity
        self.threshold_agree = threshold_agree
        self.threshold_intrigued = threshold_intrigued
        self.belief_strength_max = belief_strength_max
        self.max_belief_capacity = max_belief_capacity
        
        self.params = {
            'id':self.idx,
            'initial_exposed_ideas':exposed_ideas,
            'creativity':self.creativity,
            'threshold_agree':threshold_agree,
            'threshold_intrigued':threshold_intrigued
        }


    def update(self,idx2people,D_sn):

        self.initiate_conversation(idx2people,D_sn)
        self.update_beliefs()

    def initiate_conversation(self,idx2people,D_sn):

        possible_targets = list(D_sn.successors(self.idx))

        if possible_targets:
            weights = np.array([D_sn[self.idx][j]['weight'] for j in possible_targets])
            weights = weights/np.sum(weights)
            target_idx = np.random.choice(possible_targets,p=weights)
            target_person = idx2people[target_idx]

            idea_idx_to_discuss = random.choices(list(self.beliefs.keys()),
                weights = list(self.beliefs.values()),
                k=1
            )[0]

            target_person.discuss(idea_idx_to_discuss,self.idx,D_sn)
        else:
            print("person does not follow anyone")


    def discuss(self,idea_idx_to_discuss,speaker_idx, D_sn):

        idx2ideas = self.ecosystem.get_idx2ideas()
        
        idea_to_discuss = idx2ideas[idea_idx_to_discuss]
        max_idea_idx = max(list(self.beliefs.keys()), key = lambda idx: idx2ideas[idx].similarity(idea_to_discuss))

        assert max_idea_idx in self.beliefs.keys()

        max_idea = idx2ideas[max_idea_idx]
        similarity = idx2ideas[max_idea_idx].similarity(idea_to_discuss)

        #decide how to respond
        if similarity>self.threshold_agree:
            
            #update belief
            self.strengthen_belief(max_idea.idx,change_in_belief=1)
            
            #update relationship strength
            D_sn[speaker_idx][self.idx]['weight'] += 1 #or rate of agreement

        else:
            if similarity>self.threshold_intrigued:
                if random.random() < self.creativity:

                    #create crossover idea
                    new_idea_bitstring = max_idea.crossover(idea_to_discuss,2)
                    new_idea_idx = self.ecosystem.add_idea(new_idea_bitstring)

                    #replace belief
                    self.replace_belief(max_idea.idx,new_idea_idx)

                    #update relationship strength
                    D_sn[speaker_idx][self.idx]['weight'] += 0.5 #half strength for half the idea
                else:
                    print("not creatived")


    def update_beliefs(self):

        percent_decrease_each_time_step = 0.02
        for belief in self._beliefs:
            self._beliefs[belief] *= 1-percent_decrease_each_time_step
    
    def strengthen_belief(self,idea_idx,change_in_belief = 1):

        if sum(list(self._beliefs.values())) < self.max_belief_capacity:
            self._beliefs[idea_idx] += change_in_belief
            
            if self._beliefs[idea_idx]>self.belief_strength_max:
                self._beliefs[idea_idx] = self.belief_strength_max

    def replace_belief(self, old_idea_idx, new_idea_idx):

        conviction = self._beliefs[old_idea_idx]
        self._beliefs.pop(old_idea_idx)
        self._beliefs[new_idea_idx] = conviction

    def get_all_params(self):
        return self.params

    @property
    def beliefs(self):
        return self._beliefs