from __future__ import annotations

from os import get_terminal_size, setregid
import random

from numpy import histogram
import numpy as np
from tools.move_choice import get_relative_move_coordinates, get_move_choice_params

from location import Location

import logging

class Agent:

    """Agent object
    """

    def __init__(
        self,
        idx: int,
        location: Location,
        job =None,
        pref_pop_density = None,
        **kwargs):

        self.idx = idx
        self.job = job
        self._location = location
        self._location.add_agent(self)
        m,b = 1.25,6
        self._score2moveprob = lambda s: 1/(1+np.exp(-(m*s-b)))

        self.pref_pop_density = random.randint(0,100) if not pref_pop_density else pref_pop_density

    @property
    def income(self):
        return self.job.salary

    def decide_to_move(self):

        coeff = 0.01
        score_housing_cost = coeff*self.location.housing_cost()/self.income

        #### calculate total score ####
        total_score =  score_housing_cost # + score_income + score_low_income + score_income_match +

        score_dict = {
            'total_score': total_score,
            'score_housing_cost': score_housing_cost
        }

        decision = (random.random() < self._score2moveprob(total_score))

        return decision, score_dict


    def decide_where_to_move(self,all_locations,**kwargs):
        
        # if 'move_choice_params' in kwargs:
        #     mcp = kwargs['move_choice_params']
        #     if isinstance(mcp,callable):
        #         num_choices, move_size_exp = mcp(self.income)
        #     else:
        #         num_choices, move_size_exp = mcp
        # else:
        #     num_choices, move_size_exp = 10,3

        choices = get_relative_move_coordinates(*get_move_choice_params(self.income))

        # get possible locations to move
        N, m_ = all_locations.shape
        x0,y0 = self.location.coords
        possible_locations = [all_locations[(x0+x)%N][(y0+y)%N] for x,y in choices]

        possible_location_scores = [self.score_location(l) for l in possible_locations]
        total_scores = [s['total_score'] for s in possible_location_scores]
        
        new_location = possible_locations[np.argmax(total_scores)]
        old_location = self.location

        return old_location,new_location, possible_location_scores

    def score_location(self,location):
        
        coeff_pop_dens = 1
        coeff_job_opp = 3
        coeff_median_income = 5*10**(-4)
        coeff_housing_cost = 0.01

        # does the location align with agents preferred population density
        score_pop_dens = -coeff_pop_dens*abs(location.capacity-self.pref_pop_density)

        # job opportunity is higher in cities
        score_job_opp = 0 if self.job.remote_status else coeff_job_opp*location.capacity

        # mismatch in income lowers score
        score_median_income = coeff_median_income*abs(location.median_income()-self.income)

        # score housing cost
        score_housing_cost = -coeff_housing_cost*self.location.housing_cost()/self.income
        
        ###print("location scores", score_pop_dens,score_job_opp,score_median_income)

        total_score = score_pop_dens + score_job_opp + score_median_income + score_housing_cost

        scoredict = {
            'total_score':total_score,
            'score_pop_dens':score_pop_dens,
            'score_job_opp':score_job_opp,
            'score_median_income':score_median_income,
            'score_housing_cost':score_housing_cost
        }
        return scoredict

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self,new_location):
        self._location= new_location

    def get_tidy_state(self):
        return self.idx, self.location.idx, *self.location.coords
        
