from __future__ import annotations

from os import get_terminal_size, setregid
import random

from numpy import histogram

from location import Location

class Agent:

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

    def decide_to_move(self):

        if random.random() < 0.1:
            return True
        return False

    def decide_where_to_move(self,all_locations):
        
        old_location = self.location
        new_location = random.choice(all_locations.flatten())

        return old_location,new_location

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self,new_location):
        self._location= new_location

    def get_tidy_state(self):
        return self.idx, self.location.idx, *self.location.coords
        
