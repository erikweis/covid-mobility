import streamlit as st
import numpy as np
import inequality
import shutil
import os

from mobilitysimulation import MobilitySimulation
from analyze import SimulationAnalysis


def get_markdown_text(fname):
    md_dir = 'streamlit_app_data/text/'
    return open(md_dir+fname).read()

st.title('Modeling the Effect of COVID-19 on Spatial Mobility')

with st.expander('Introduction'):
    st.markdown(get_markdown_text('intro.md'))

with st.expander('Emperical Motivations'):
    st.markdown(get_markdown_text('empirical_motivations.md'))

with st.expander('Model Summary'):
    st.markdown(get_markdown_text('model_summary.md'))

with st.expander('Results'):
    st.markdown(get_markdown_text('results.md'))


st.sidebar.slider('Grid Size',min_value=10,max_value=30,value=20,key='grid_size')
st.sidebar.slider('Number of Agents',min_value=st.session_state.grid_size**2,max_value=5000,value=3000,step=50,key='num_agents')
T = st.sidebar.slider('Number of Time Steps',min_value=100,max_value=1000,value=500,step=50,key='num_steps')
st.sidebar.slider('Covid Intervention Time',min_value=0,max_value=st.session_state.num_steps,value=int(T/2),step=5,key='intervention_time')

weights = {
        'pop_dens':1,
        'job_opp':1,
        'median_income':1,
        'housing_cost':1
    }

def generate_new_data():

    dirname = 'streamlit_app_data/simulation_data'
    shutil.rmtree(dirname)
    os.mkdir(dirname)

    m = MobilitySimulation(
        dirname,
        grid_size=st.session_state.grid_size,
        num_steps = st.session_state.num_steps,
        num_agents = st.session_state.num_agents,
        covid_intervention_time = st.session_state.intervention_time,
        income_distribution = 'power',
        location_score_weights=weights,
        total_occupancy = 0.9
    )
    m.run_simulation()
    m.save_data()

    a = SimulationAnalysis(dirname='streamlit_app_data/simulation_data',override_plots=True)
    a.plot_flows()
    a.plot_move_activity_by_income(smoothing=True)
    a.plot_occupancy_pre_post_covid()
    
def get_image_path(fname):
    dirname = 'streamlit_app_data/simulation_data/'
    return dirname + fname

st.sidebar.button('Run Simulation',key='run_button',on_click=generate_new_data)
st.image(get_image_path('flows.png'))
st.image(get_image_path('move_activity_by_income_bracket.png'))
st.image(get_image_path('occupancy_pre_post_covid.png'))