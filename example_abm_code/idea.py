import random
import secrets

from bitarray import bitarray
from bitarray.util import count_xor

class Idea:

    def __init__(self,idx=None, bitstring=None, length=8):

        self.idx = idx 
        self._bitstring = bitarray(bitstring)
        if not self._bitstring:
            self._bitstring = bitarray(''.join(str(round(random.random())) for _ in range(length)))

    @property
    def bitstring(self):
        return self._bitstring

    def similarity(self,other_idea):
        try:
            return 1- count_xor(self.bitstring, other_idea.bitstring)/len(self.bitstring)
        except:
            print(self.bitstring,other_idea.bitstring)
    def crossover(self,other_idea,num_points):

        index_list = list(range(1,len(other_idea)-1)) #cant have crossover at first/last index
        crossover_points = [0]+sorted(random.sample(index_list,num_points))+[len(other_idea)]

        ideas = [self.bitstring.to01(), other_idea.bitstring.to01()]
        idea_choice = round(random.random()) #0 or 1
        new_idea = ''

        for i in range(0,num_points+1):
            start, end = crossover_points[i:i+2] #get indicies to slice
            new_idea += ideas[idea_choice][start:end]
            idea_choice = (idea_choice + 1)%2

        assert len(new_idea)==len(other_idea)

        try: 
            bitarray(new_idea)
        except:
            print(new_idea,*ideas)
            print(str(ideas[0]))

        return new_idea

    def __len__(self):
        return len(self.bitstring)


if __name__ == "__main__":

    random.seed(5)

    x = Idea(idx='x')
    y = Idea(idx='y')

    print(x.bitstring)
    print(y.bitstring)
    print(x.similarity(y))
    print(x.crossover(y,2))