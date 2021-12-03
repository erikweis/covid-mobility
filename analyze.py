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
from tqdm import tqdm

from matplotlib.image import AxesImage

class SimulationAnalysis:

    def __init__(self, experiment_name, folder_name):

        self.dirname = os.path.join('data',experiment_name,folder_name)
        
        # load agent locationd ata
        self.df = pd.read_csv(os.path.join(self.dirname,'agent_location_data.csv'))

        # load move data
        self.move_df = pd.read_csv(os.path.join(self.dirname,'move_data.csv'))

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

        #join agent and move_df
        self.merged_move_df = pd.merge(self.move_df, self.agents_df, on='agentID')
        self.merged_move_df['income_bracket'] = pd.qcut(self.merged_df['income'], 
                                                    q=5,
                                                    labels=['low','lower_mid','mid','upper_mid', 'high'])

    def check_existance(self,fname):

        if os.path.isfile(os.path.join(self.dirname,fname)):
            print(f"File {fname} already exists")
            return True
        return False


    def animate(self, animation_data, fout_name, plot_title = None):
        
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


    def multi_animate(self,data_array,fout_name, plot_titles = None,figsize=(8,8)):

        num_rows,num_cols = data_array.shape[:2]

        fig, axes = plt.subplots(*data_array.shape[:2],figsize=figsize)

        def mapping(t):
            
            ims = np.empty((num_rows,num_cols),dtype=AxesImage)
            
            #only rows
            if num_rows == 1:
                i=0
                for j in range(num_cols):
                    axes[i][j].set_title(plot_titles[i][j])
                    ims[i][j] = axes[j].imshow(data_array[i][j][t])

            #only columns
            elif num_cols == 1:
                j=0
                for i in range(num_rows):
                    axes[i][j].set_title(plot_titles[i][j])
                    ims[i][j] = axes[i].imshow(data_array[i][j][t])

            # multiple rows and columns
            else:
                for i in range(num_rows):
                    for j in range(num_cols):
                        axes[i][j].set_title(plot_titles[i][j])
                        ims[i][j] = axes[i][j].imshow(data_array[i][j][t])
                
            return ims

        ims = mapping(0)
        for ax in axes.flatten():
            ax.set_axis_off()
        fig.colorbar(ims.flatten()[-1], ax=axes, shrink=0.6)

        def update(t):
            images = mapping(t)
            return images.flatten()

        anim = animation.FuncAnimation(fig, update,interval=100,frames=self.num_steps,repeat=True)

        f = os.path.join(self.dirname,fout_name)
        #writergif = animation.PillowWriter(fps=20) 
        writervideo = animation.FFMpegWriter(fps=10) 
        anim.save(f, writer=writervideo)
        

    def get_data_for_animation_population_density(self, df):

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size),dtype=int)

        for index, row in tqdm(df.iterrows()):
            t = row['time']
            x = row['xloc']
            y = row['yloc']
            data[t][y][x] += 1

        return data

    
    def get_data_for_animation_median_income(self):

        left = self.merged_df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        df = pd.merge(left,self.locations_df,on='locationID')

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size))

        for index, row in df.iterrows():
            t = row['time']
            x,y = row['coords']
            data[t][y][x] = row['Median_Income']

        return data


    def get_data_for_animation_std_income(self):

        left = self.merged_df.groupby(['time','locationID'])['income'].std().reset_index(name='std_income')
        df = pd.merge(left,self.locations_df,on='locationID')

        data = np.zeros((self.num_steps,self.grid_size,self.grid_size))

        for index, row in df.iterrows():
            t = row['time']
            x,y = row['coords']
            data[t][y][x] = row['std_income']

        return data


    def animate_population_density_by_salary(self):

        filename = "income_bracket_animation.mov"
        if self.check_existance(filename):
            return None

        data_array = []
        plot_titles = []
        for income_bracket in self.merged_df['income_bracket'].unique():
            df = self.merged_df
            data = self.get_data_for_animation_population_density(df[df['income_bracket'] == income_bracket])
            data_array.append(data)
            plot_titles.append(income_bracket)
        
        #add total as sixth plot
        data_array.append(self.get_data_for_animation_population_density(self.merged_df))
        plot_titles.append('all agents')

        data_array = np.array(data_array)
        data_array = data_array.reshape((3,2,*data_array.shape[1:]))
        plot_titles = np.array(plot_titles).reshape((3,2))
        self.multi_animate(data_array, filename,plot_titles = plot_titles,figsize=(8,10))


    def animate_population_density(self):

        filename = 'population_density_animation.mov'
        if os.path.isfile(os.path.join(self.dirname,filename)):
            print("file already created")
            return None

        data = self.get_data_for_animation_population_density(self.merged_df)
        self.animate(data,filename)

 
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



    def calculate_move_distances(self):

        df = self.move_df
        print(self.move_df.columns)

        def calc_distance(row):
            N = self.grid_size

            x1,y1 = row['from_x'], row['from_y']
            x2,y2 = row['to_x'], row['to_y']

            x_dist = min(abs(x1+N-x2),abs(x1-(x2+N)),abs(x1-x2))
            y_dist = min(abs(y1+N-y2),abs(y1-(y2+N)),abs(y1-y2))

            return np.sqrt(x_dist**2 + y_dist**2)

        df['move_distance'] = df.apply(calc_distance,axis=1)
        print(df['move_distance'])


    def plot_move_distance_distribution(self):

        # get move distances
        filename = 'move_distance_distribution.png'
        if self.check_existance(filename):
            return None

        if 'move_distance' not in self.move_df.columns:
            self.calculate_move_distances()
        move_distances = self.move_df['move_distance'].values

        bins = np.logspace(np.log10(np.min(move_distances+0.01)), 
                       np.log10(np.max(move_distances)), 
                       num=10)
        vals, bins_ = np.histogram(move_distances,bins=bins)

        plt.plot(bins[:-1],vals,'o-')
        plt.title('Move Distance Distribution')
        plt.savefig(os.path.join(self.dirname,filename))


    def plot_move_activity_by_income(self):

        filename = 'move_activity_by_income_bracket.png'
        if self.check_existance(filename):
            return None

        df = self.merged_move_df
        df = df.groupby(['income_bracket','time']).size().reset_index(name='Count')
        df = pd.pivot_table(df, index='time',columns = 'income_bracket', values='Count')
        
        #plot
        fig, ax = plt.subplots()
        for col in df.columns:
            ax.plot(df[col],label=col)
        ax.legend()
        ax.set_title('Move Activity by Income Bracket')
        plt.savefig(os.path.join(self.dirname,filename))


if __name__ == "__main__":
    
    foldername = 'trial_0'
    experiment_name = 'movement2'

    if not foldername:
        folders = [f for f in os.listdir('data/') if not f.startswith('.')]
        foldername = max(folders,key=lambda f: datetime.strptime(f,"%m-%d_%H-%M-%S"))
        print("Analyzing folder {}".format(foldername))

    sa = SimulationAnalysis(experiment_name, foldername)
    print("loaded")
    sa.plot_move_distance_distribution()
    #sa.animate_population_density()
    sa.animate_population_density_by_salary()
    sa.plot_move_activity_by_income()
    #sa.plot_location_demographics(0)
    #sa.plot_median_incomes()
    #sa.plot_income_std()

    #sa.animate_median_income()
    #sa.animate_std_income()