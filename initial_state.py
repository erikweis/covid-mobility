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
    scaling_factor = (total_occupancy-N**2)/np.sum(vals.flatten()) #normalize for total_occupancy - N^2
    capacities = vals*scaling_factor
    capacities = capacities + np.ones((N,N)) #add remaining N^2 evenly across board

    return np.round(capacities).astype(int)


def plot_capacities(caps):

    plt.imshow(caps)
    plt.colorbar()
    plt.show()


def plot_capacity_distribution(caps):

    caps = caps.flatten()

    out,bins = np.histogram(caps,bins=np.logspace(np.log10(0.1),np.log10(100), 20),density=True)

    plt.scatter(bins[:-1],out)
    plt.yscale('log')
    plt.show()

if __name__ == "__main__":

    caps = get_initial_capacities(30,50,5000)

    plot_capacities(caps)
    plot_capacity_distribution(caps)
    

