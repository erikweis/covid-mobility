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

        """We want a function that determines housing cost from two variables:
        1) median income and 2) the demand. We use occupancy as a proxy for demand."""
        
        if self.occupancy_rate() >= 1:
            return 1000000 #very high value if too many people live there
        else:
            
            base_cost = 0.2*self.median_income() # a reasonable proportion of median income
            demand_adjustment = 1500*self.occupancy_rate()/(1-self.occupancy_rate()) # depends on occupancy

            return base_cost + demand_adjustment
