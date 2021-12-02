import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

        with open(os.path.join(self.dirname,'locations.jsonl')) as f:
            self.locations_df = pd.DataFrame([json.loads(s) for s in f.readlines()])
            self.locations_df.rename(columns={'idx':'locationID'},inplace=True)

        # Join agent and data.csv on idx
        self.merged_df = pd.merge(self.df, self.agents_df, on='agentID')
        self.merged_df['income_bracket'] = pd.qcut(self.merged_df['income'], 
                                                    q=5,
                                                    labels=['low','lower_mid','mid','upper_mid', 'high'])

    def get_data_for_animation_by_density(self, df):

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size),dtype=int)

        for index, row in df.iterrows():
            t = row['time']
            x = row['xloc']
            y = row['yloc']
            data[t][y][x] += 1

        return data


    def animate(self, animation_data, fout_name):
        
        fig, ax = plt.subplots()

        im = ax.imshow(animation_data[0])
        ax.set_title('0')
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', '5%', '5%')
        fig.colorbar(im,cax=cax)

        def update(i):
            im.set_array(animation_data[i])
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

        filename = 'location_animation.mov'
        if os.path.isfile(os.path.join(self.dirname,filename)):
            print("file already created")
            return None

        data = self.get_data_for_animation_by_density(self.merged_df)
        self.animate(data,filename)
        

    def get_data_for_animation_median_income(self):

        left = self.merged_df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        df = pd.merge(left,self.locations_df,on='locationID')

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size))

        for index, row in df.iterrows():
            t = row['time']
            x,y = row['coords']
            data[t][y][x] = row['Median_Income']

        return data

        #df = pd.pivot_table(df,index='time',columns='locationID',values='Median_Income')

    def get_data_for_animation_std_income(self):

        left = self.merged_df.groupby(['time','locationID'])['income'].std().reset_index(name='std_income')
        df = pd.merge(left,self.locations_df,on='locationID')

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size))

        for index, row in df.iterrows():
            t = row['time']
            x,y = row['coords']
            data[t][y][x] = row['std_income']

        return data

    def animate_median_income(self):

        filename = "median_income_animation.mov"
        if os.path.isfile(os.path.join(self.dirname,filename)):
            print("file already created")
            return None

        data = self.get_data_for_animation_median_income()
        self.animate(data,filename)

    def animate_std_income(self):

        filename = "std_income_animation.mov"
        if os.path.isfile(os.path.join(self.dirname,filename)):
            print("file already created")
            return None

        data = self.get_data_for_animation_std_income()
        self.animate(data,filename)


    def plot_location_demographics(self,locationID):

        df = self.merged_df[self.merged_df['locationID']==locationID]
        df = df.groupby(['income_bracket','time']).size().reset_index(name='Count')
        df = pd.pivot_table(df, index='time',columns = 'income_bracket', values='Count')
        
        fig, ax = plt.subplots()
        
        for col in df.columns:
            ax.plot(df[col],label=col)
        
        ax.legend()
        ax.set_title(f'Location {locationID}')
        plt.show()


    def get_location_median_incomes(self):

        """Returns dataframe with each column as the time-dependent median
        income of a particular location. All locations are represented.
        """

        df = self.merged_df
        df = df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        return pd.pivot_table(df,index='time',columns='locationID',values='Median_Income')


    def plot_median_incomes(self):

        df = self.get_location_median_incomes()
        for col in df.columns:
            plt.plot(df[col],color='blue',alpha=0.1)
        plt.show()
        
    def get_income_std(self):
        
        """[summary]
        """

        df = self.merged_df
        df = df.groupby(['time','locationID'])['income'].std().reset_index(name='Income_Std')
        return pd.pivot_table(df,index='time',columns='locationID',values='Income_Std')

    def plot_income_std(self):

        df = self.get_income_std()
        for col in df.columns:
            plt.plot(df[col],color='blue',alpha=0.1)
        plt.show()


if __name__ == "__main__":
    
    foldername = 'trial_0'
    experiment_name = 'movement3'

    if not foldername:
        folders = [f for f in os.listdir('data/') if not f.startswith('.')]
        foldername = max(folders,key=lambda f: datetime.strptime(f,"%m-%d_%H-%M-%S"))
        print("Analyzing folder {}".format(foldername))

    sa = SimulationAnalysis(experiment_name, foldername)
    print("loaded")
    sa.animate_location_density()
    # #sa.animate_location_by_salary()
    #sa.plot_location_demographics(0)
    #sa.plot_median_incomes()
    #sa.plot_income_std()

    sa.animate_median_income()
    sa.animate_std_income()