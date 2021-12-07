from __future__ import annotations

from os import get_terminal_size, setregid
import random

from numpy import histogram
import numpy as np
from scipy.special import expit

from location import Location

import logging

LOG_PROBABILITY = 0.0001

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

        #### define coefficients ####
        # set coefficients such that each factor (on average) contributes equall
        # to the overall score
        coeff_income = 10**(-10)
        coeff_low_income = 10
        coeff_income_match = 10**(-5)
        coeff_housing_cost = 10**(-4)

        #### consider various factors ####

        # higher income means likely to move farther
        score_income = coeff_income*(self.income**2)

        # very low income also means likely to move (getting kicked out)
        s = 1-coeff_low_income*(1/(self.income+1))
        score_low_income = 0 if s<0 else s

        # if living below means (in a rich area) or above means (a poor area), more likely to move
        score_income_match = coeff_income_match*abs(self.income-self.location.median_income())

        # housing cost
        score_housing_cost = coeff_housing_cost*(self.location.housing_cost()**4)

        #### calculate total score ####
        total_score = score_income + score_low_income + score_income_match + score_housing_cost

        score_dict = {
            'total_score': total_score,
            'score_income': score_income,
            'score_income_match': score_income_match,
            'score_housing_cost': score_housing_cost
        }

        # normalization should be set such that the expected move rate overall matches
        # emperical data
        
        decision = (random.random() < self._score2moveprob(total_score))

        return decision, score_dict


    def decide_where_to_move(self,all_locations):
        
        # higher salary people have the resources to search more places
        number_of_choices = 10 #function depending on salary

        # larger power law exponent (a), where x^(1/a), for higher income
        # higher income people have the resources to move further
        a = 2

        #create random x and random y drawn from power law distribution
        xs = np.random.zipf(3, size=number_of_choices)
        ys = np.random.zipf(3, size=number_of_choices)
        #random sign to account for moving left and right, up and down
        signs_x = np.random.choice([-1,1],size=number_of_choices)
        signs_y = np.random.choice([-1,1],size=number_of_choices)
        xs = np.floor(xs*signs_x).astype(int)
        ys = np.floor(ys*signs_y).astype(int)

        #add current location, if it's the highest score don't move
        xs = np.append(xs,[0]); ys = np.append(ys,[0]);

        # reference coordinates
        N, M = all_locations.shape
        x0,y0 = self.location.coords

        # get possible locations to move
        possible_locations = [all_locations[(x0+x)%N][(y0+y)%N] for x,y in zip(xs,ys)]

        possible_location_scores = [self.score_location(l) for l in possible_locations]
        total_scores = [s['total_score'] for s in possible_location_scores]
        
        new_location = possible_locations[np.argmax(total_scores)]
        old_location = self.location

        #new_location = random.choice(possible_locations)
        return old_location,new_location, possible_location_scores

    def score_location(self,location):
        
        coeff_pop_dens = 1
        coeff_job_opp = 1
        coeff_median_income = 10**(-4)
        coeff_housing_cost = 10

        # does the location align with agents preferred population density
        score_pop_dens = -coeff_pop_dens*abs(location.capacity-self.pref_pop_density)

        # job opportunity is higher in cities
        score_job_opp = 0 if self.job.remote_status else coeff_job_opp*location.capacity

        # mismatch in income lowers score
        score_median_income = coeff_median_income*abs(location.median_income()-self.income)

        # score housing cost
        score_housing_cost = -coeff_housing_cost*self.location.housing_cost()
        
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
        
