import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import os
import json

class SimulationAnalysis:

    def __init__(self, experiment_name, folder_name):

        self.dirname = os.path.join('data',experiment_name,folder_name)
        
        #data frames from files
        self.df = pd.read_csv(os.path.join(self.dirname,'data.csv'))

        with open(os.path.join(self.dirname,'params.json'),'r') as f:
            params =json.load(f)

        for attr,value in params.items():
            self.__setattr__(attr,value)

        #parse jsonl agent locations
        with open(os.path.join(self.dirname,'agents.jsonl')) as f:
            self.agents_df = pd.DataFrame([json.loads(s) for s in f.readlines()])
            self.agents_df.rename(columns={'idx': 'agentID'}, inplace=True)

        # Join agent and data.csv on idx
        self.merged_df = pd.merge(self.df, self.agents_df, on='agentID')
        self.merged_df['income_bracket'] = pd.qcut(self.merged_df['income'], 
                                                    q=5,
                                                    labels=['low','lower_mid','mid','upper_mid', 'high'])


    def animation_boilerplate(self, df, fout_name):

        if os.path.isfile(os.path.join(self.dirname, fout_name)):
            print("file already created")
            return None

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size),dtype=int)

        for index, row in df.iterrows():
            t = row['time']
            x = row['xloc']
            y = row['yloc']
            data[t][y][x] += 1

        fig, ax = plt.subplots()

        im = ax.imshow(data[0])
        ax.set_title('0')

        def update(i):
            im.set_array(data[i])
            ax.set_title(f'Frame {i}')
            return im,

        anim = animation.FuncAnimation(fig, update,interval=100,frames=self.num_steps,repeat=True)

        f = os.path.join(self.dirname,fout_name)
        #writergif = animation.PillowWriter(fps=20) 
        writervideo = animation.FFMpegWriter(fps=10) 
        anim.save(f, writer=writervideo)


    def animate_location_by_salary(self):
        for income_bracket in self.merged_df['income_bracket'].unique():
            df = self.merged_df
            self.animation_boilerplate(df[df['income_bracket'] == income_bracket], f"income_animation_{income_bracket}.mov")


    def animate_location_density(self):

        if os.path.isfile(os.path.join(self.dirname,'location_animation.mov')):
            print("file already created")
            return None

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size),dtype=int)

        for index, row in self.df.iterrows():
            t = row['time']
            x = row['xloc']
            y = row['yloc']
            data[t][y][x] += 1

        fig, ax = plt.subplots()

        im = ax.imshow(data[0])
        ax.set_title('0')

        def update(i):
            im.set_array(data[i])
            ax.set_title(f'Frame {i}')
            return im,

        anim = animation.FuncAnimation(fig, update,interval=100,frames=self.num_steps,repeat=True)

        f = os.path.join(self.dirname,'location_animation.mov')
        #writergif = animation.PillowWriter(fps=20) 
        writervideo = animation.FFMpegWriter(fps=10) 
        anim.save(f, writer=writervideo)


if __name__ == "__main__":
    
    foldername = 'trial_0'
    experiment_name = 'movement1'

    if not foldername:
        folders = [f for f in os.listdir('data/') if not f.startswith('.')]
        foldername = max(folders,key=lambda f: datetime.strptime(f,"%m-%d_%H-%M-%S"))
        print("Analyzing folder {}".format(foldername))

    sa = SimulationAnalysis(experiment_name, foldername)
    print("loaded")
    # sa.animate_location_density()
    sa.animate_location_by_salary()