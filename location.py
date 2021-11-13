class Location:

    def __init__(self,x,y,capacity):

        self.x = x
        self.y = y
        self.capacity = capacity
        self.agents = None


    def occupancy_rate(self):
        return len(self.agents)/self.capacity