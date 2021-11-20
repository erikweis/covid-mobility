class Location:

    def __init__(
        self,
        idx,
        coords = None,
        capacity=None,
        agents=None
    ):

        self.idx = idx
        self.coords = coords
        self.capacity = capacity
        if not agents:
            self.agents = []

    def add_agent(self,agent):
        self.agents.append(agent)

    def remove_agent(self,agent):
        self.agents.remove(agent)

    def occupancy_rate(self):
        return len(self.agents)/self.capacity