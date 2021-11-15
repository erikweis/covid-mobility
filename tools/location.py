
class Location:
    """
    Just a placeholder for purposes of working on the grid initialization and
    visualization methods.
    """
    def __init__(self,
                 capacity: int = 0):
        self._capacity = capacity

    def get_capacity(self):
        return self._capacity

    def set_capacity(self, value):
        self._capacity = value
