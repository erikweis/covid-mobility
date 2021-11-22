import pytest
from mobilitysimulation import MobilitySimulation
from location import Location
from agent import Agent

def test_mobilitysimulation_initialization():

    num_agents = 10
    grid_size = 12

    ms = MobilitySimulation(
        'not_needed',
        num_agents = num_agents,
        grid_size=grid_size
    )

   #correct grid size
    assert ms.locations.shape == (grid_size,grid_size)

    #check that all elements of locations are Location objects
    assert all(isinstance(loc,Location) for loc in ms.locations.flatten())

    #check that all agents are in a location and that no agents are in multiple locations
    agents_from_locations = []
    for location in ms.locations.flatten():
        agents_from_locations += location.agents
    assert len(agents_from_locations) == len(ms.agents)

    
    
