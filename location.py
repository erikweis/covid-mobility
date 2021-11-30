import numpy as np

class Location:

    def __init__(
        self,
        idx,
        coords = None,
        capacity=None,
        agents=None,
        quality=None
    ):

        self.idx = idx
        self.coords = coords
        self.capacity = capacity
        self.quality = quality
        if not agents:
            self.agents = []

    def add_agent(self,agent):
        self.agents.append(agent)

    def remove_agent(self,agent):
        self.agents.remove(agent)

    def median_income(self):
        return np.mean([a.income for a in self.agents])

    def occupancy_rate(self):
        return len(self.agents)/self.capacity

    def housing_cost(self):
        return self.occupancy_rate()
