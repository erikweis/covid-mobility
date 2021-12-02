import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def get_initial_capacities(grid_size, num_cities,total_occupancy):
    
    N = grid_size
    x, y = np.mgrid[0:N:1, 0:N:1]
    pos = np.dstack((x, y))

    vals = np.zeros((N,N))
    for _ in range(num_cities):
        xloc,yloc = np.random.randint(N,size=2)
        size = np.random.zipf(2)
        vals += size*multivariate_normal([xloc,yloc],[[size, 0], [0, size]]).pdf(pos)
    
    #TODO wrap around by creating a city with location in middle of grid and shifting to location

    #blanket low occupancy for rural areas
    total_rural_pop_to_add = sum(vals.flatten())*0.1
    vals += total_rural_pop_to_add/(N**2)

    #normalize
    scaling_factor = total_occupancy/np.sum(vals.flatten())
    capacities = vals*scaling_factor

    return np.round(capacities).astype(int)


if __name__ == "__main__":

    caps = get_initial_capacities(30,50,5000)
    plt.imshow(caps)
    plt.colorbar()
    plt.show()