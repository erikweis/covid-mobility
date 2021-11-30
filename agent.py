from __future__ import annotations

from os import get_terminal_size, setregid
import random

from numpy import histogram
import numpy as np

from location import Location

class Agent:

    """Agent object
    """

    def __init__(
        self,
        idx: int,
        location: Location,
        job =None,
        **kwargs):

        self.idx = idx
        self.job = job
        self._location = location
        self._location.add_agent(self)

    @property
    def income(self):
        return self.job.salary

    def decide_to_move(self):

        #### define coefficients ####
        # set coefficients such that each factor (on average) contributes equall
        # to the overall score
        coeff_income = 0.01
        coeff_low_income = 10
        coeff_income_match = 0.05
        coeff_housing_cost = 1

        #### consider various factors ####

        # higher income means likely to move farther
        score_income = coeff_income*(self.income**2)

        # very low income also means likely to move (getting kicked out)
        s = 1-coeff_low_income*(1/self.income)
        score_low_income = 0 if s<0 else s

        # if living below means (in a rich area) or above means (a poor area), more likely to move
        score_income_match = coeff_income_match*abs(self.income-self.location.get_median_income())

        # housing cost
        score_housing_cost = coeff_housing_cost*self.location.housing_cost()

        #### calculate total score ####
        total_score = score_income + score_low_income + score_income_match + score_housing_cost
        normalization = 100 
        p_move = total_score/normalization 
        # normalization should be set such that the expected move rate overall matches
        # emperical data
        
        return (random.random() < total_score/normalization)

    def decide_where_to_move(self,all_locations):
        
        # higher salary people have the resources to search more places
        number_of_choices = None #function depending on salary

        # larger power law exponent (a), where x^(1/a), for higher income
        # higher income people have the resources to move further
        a = None

        #create random x and random y drawn from power law distribution
        xs = np.random.power(1/a, size=number_of_choices)
        ys = np.random.power(1/a, size=number_of_choices)
        #random sign to account for moving left and right, up and down
        signs_x = np.random.choice([-1,1],size=number_of_choices)
        signs_y = np.random.choice([-1,1],size=number_of_choices)
        xs = np.round(xs*signs_x)
        ys = np.round(ys*signs_y)

        # reference coordinates
        N, M = all_locations.shape
        x0,y0 = self.location.coords

        # get possible locations to move
        possible_locations = [all_locations[(x0+x)%N][(y0+y)%N] for x,y in zip(xs,ys)]

        scores = [self.score_location(l) for l in possible_locations]
        
        new_location = possible_locations[np.argmax(scores)]
        old_location = self.location

        #new_location = random.choice(possible_locations)

        return old_location,new_location

    def score_location(self,location):
        pass

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self,new_location):
        self._location= new_location

    def get_tidy_state(self):
        return self.idx, self.location.idx, *self.location.coords
        
