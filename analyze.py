import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import datetime
import os
import json
from tqdm import tqdm
import time

from seaborn import kdeplot

from matplotlib.image import AxesImage

class SimulationAnalysis:

    def __init__(self, experiment_name, folder_name):

        #the working directory of this particular simulation
        self.dirname = os.path.join('data',experiment_name,folder_name)

        #set all meta parameters to object attributes
        with open(os.path.join(self.dirname,'params.json'),'r') as f:
            params =json.load(f)
        for attr,value in params.items():
            self.__setattr__(attr,value)

        #parse jsonl agent locations
        with open(os.path.join(self.dirname,'agents.jsonl')) as f:
            self.agents_df = pd.DataFrame([json.loads(s) for s in f.readlines()])
            self.agents_df.rename(columns={'idx': 'agentID'}, inplace=True)
            self.agents_df['income_bracket'] = pd.qcut(
                self.agents_df['income'], 
                q=5,
                labels=['low','lower_mid','mid','upper_mid', 'high'])

        with open(os.path.join(self.dirname,'locations.jsonl')) as f:
            self.locations_df = pd.DataFrame([json.loads(s) for s in f.readlines()])
            self.locations_df.rename(columns={'idx':'locationID'},inplace=True)
        

    #### lazy load data frames, i.e. they only load the first time the argument is called ####

    @property
    def df(self):
        # load agent locationd ata
        if not hasattr(self,'_df'):
            self._df = pd.read_csv(os.path.join(self.dirname,'agent_location_data.csv'),dtype={'xloc':int,'yloc':int})
        return self._df

    @property
    def move_df(self):
        # load move data
        if not hasattr(self,'_move_df'):
            self._move_df = pd.read_csv(os.path.join(self.dirname,'move_data.csv'),dtype={'time':int})
        return self._move_df
    
    @property
    def possible_location_score_df(self):
        # load location score data
        if not hasattr(self,'_possible_location_score_df'):
            self._possible_location_score_df = pd.read_csv(os.path.join(self.dirname,'possible_location_score_data.csv'),index_col=[0])
        return self._possible_location_score_df

    @property
    def location_score_df(self):
        if not hasattr(self,'_location_score_df'):
            self._location_score_df = pd.read_csv(os.path.join(self.dirname,'location_score_data.csv'),index_col=[0])
        return self._location_score_df

    
    @property
    def merged_df(self):

        # Join agent and data.csv on idx
        if not hasattr(self,'_merged_df'):
            self._merged_df = pd.merge(self.df, self.agents_df, on='agentID')
            self._merged_df['income_bracket'] = pd.qcut(self.merged_df['income'], 
                                                        q=5,
                                                        labels=['low','lower_mid','mid','upper_mid', 'high'])
        return self._merged_df

    @property
    def merged_move_df(self):

        if not hasattr(self,'_merged_move_df'):

            #join agent and move_df
            self._merged_move_df = pd.merge(self.move_df, self.agents_df, on='agentID')
            self._merged_move_df['income_bracket'] = pd.qcut(self.merged_df['income'], 
                                                        q=5,
                                                        labels=['low','lower_mid','mid','upper_mid', 'high'])
        return self._merged_move_df

    @property
    def move_decision_score_df(self):
        
        if not hasattr(self,'_move_decision_score_df'):
            self._move_decision_score_df = pd.read_csv(os.path.join(self.dirname,'move_decision_score_data.csv'))
        
        return self._move_decision_score_df


    def check_existance(self,fname):

        """helper function to avoid reduntanly generating plot images and videos."""

        if os.path.isfile(os.path.join(self.dirname,fname)):
            print(f"File {fname} already exists")
            return True
        return False


    @staticmethod
    def progress_callback(i,n):
        if i%10==0:
            print(f"t={i}/{n}")

    def animate(self, animation_data, fout_name, plot_title = None):
        
        """General function for animating values on a grid. Pass in a numpy array with dimension T x N x N,
        where N is self.grid_size and T is self.num_steps."""

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
        anim.save(f, writer=writervideo,progress_callback=self.progress_callback)


    def multi_animate(self,data_array,fout_name, plot_presence = None, plot_titles = None,figsize=(8,8)):

        """Pass in a (rows,cols) array of animation data arrays, each with dimension T x N x N. This
        is equivalent to passing an array of dimension (row,cols,T,N,N)

        Some special consideration is given to arrays where one dimension is 1, e.g. 3x1 or 1x2. Matplotlib 
        removes the two-dimension, so this just requires a bit of edge casing.

        Args:
            data_array:
            fout_name: filename to save the video
            plot_presence: array of shape (rows,cols) with boolean values corresponding to which plots should be active. 
                Useful for turning off axes.
            plot_titles: array of shape (rows, cols) with names of plots
        """

        num_rows,num_cols = data_array.shape[:2]

        fig, axes = plt.subplots(*data_array.shape[:2],figsize=figsize)
        
        #reshape axes so it's always two dimensional
        if num_rows == 1 or num_cols == 1:
            axes = np.array([axes])

        #empty image object array
        ims = np.empty((num_rows,num_cols),dtype=AxesImage)

        def set_data(t):
            
            
            for i in range(num_rows):
                for j in range(num_cols):
                    if plot_presence and not plot_presence[i][j]: continue
                    axes[i][j].set_title(plot_titles[i][j])

                    if t==0:
                        ims[i][j] = axes[i][j].imshow(data_array[i][j][t])
                    else:
                        ims[i][j].set_array(data_array[i][j][t])
            return ims

        ims = set_data(0)
        for ax in axes.flatten():
            ax.set_axis_off()
        for im,ax in zip(ims.flatten(),axes.flatten()):
            fig.colorbar(im,ax=ax,shrink=0.6)

        def update(t):
            images = set_data(t)
            return images.flatten()

        anim = animation.FuncAnimation(fig, update,interval=100,frames=self.num_steps,repeat=True)

        f = os.path.join(self.dirname,fout_name)
        #writergif = animation.PillowWriter(fps=20) 
        writervideo = animation.FFMpegWriter(fps=10) 
        anim.save(f, writer=writervideo,progress_callback=self.progress_callback)


    def animate_scatter(self,animation_data, fout_name, plot_title = None,xlabel=None, ylabel=None):

        """
        Pass in animation data as an array T by 2 by N.
        """

        fig, ax = plt.subplots()

        sca = ax.scatter(*np.array(animation_data[0]).transpose())
        ax.set_title(plot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        def update(i):
            if animation_data[i]:
                sca.set_offsets(animation_data[i])
            return sca,
        anim = animation.FuncAnimation(fig, update,interval=100,frames=self.num_steps,repeat=True)

        f = os.path.join(self.dirname,fout_name)
        #writergif = animation.PillowWriter(fps=20) 
        writervideo = animation.FFMpegWriter(fps=10) 
        anim.save(f, writer=writervideo,progress_callback=self.progress_callback)

    
    def get_data_for_animation_median_income(self):

        """Get data for animation. Array has values array[t][y][x] = median income of agents
        at location (x,y) at time t.

        Returns:
            numpy ndarray: array of dimension T x N x N
        """

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

        """use multi-animate fo generate the population density over time animations."""

        filename = "income_bracket_animation.mov"
        if self.check_existance(filename):
            return None

        T, N = self.num_steps, self.grid_size
        data = {k:np.zeros((T,N,N),dtype=int) for k in self.merged_df['income_bracket'].unique()}

        df = self.merged_df.groupby(['time','locationID','income_bracket']).agg({'agentID':'count','xloc':'first','yloc':'first'})
        df.reset_index(inplace=True)
        df.rename(columns={'agentID':'count'},inplace=True)

        for index, row in df.iterrows():
            ib = row['income_bracket']
            t = row['time']
            count = row['count']
            
            if count>0:
                xloc, yloc = int(row['xloc']),int(row['yloc'])
                data[ib][t][yloc][xloc] = row['count']
        
        data_array = list(data.values())
        plot_titles = list(data.keys())
        
        #add total density as sixth plot
        total = np.sum(d for d in data_array)
        data_array.append(total)

        plot_titles.append('all agents')

        # #reshape data into a 3x2 grid
        data_array = np.array(data_array)
        data_array = data_array.reshape((3,2,*data_array.shape[1:]))
        plot_titles = np.array(plot_titles).reshape((3,2))
        
        # #call multi-animate
        self.multi_animate(data_array, filename,plot_titles = plot_titles,figsize=(8,10))


    def animate_population_density(self):

        """animate population density"""

        filename = 'population_density_animation.mov'
        if self.check_existance(filename):
            return None

        data = self.get_data_for_animation_population_density(self.merged_df)
        self.animate(data,filename,plot_title='Population Density')


    def animate_occupancy(self):

        filename = 'occupancy_animation.mov'
        if self.check_existance(filename):
            return None

        data = self.get_data_for

 
    def animate_median_income(self):

        """animate median income at each location over time"""

        filename = "median_income_animation.mov"
        if self.check_existance(filename):
            return None

        data = self.get_data_for_animation_median_income()
        self.animate(data,filename,plot_title = 'Median Income')


    def animate_std_income(self):

        """animate standard deviation of the incomes of all agents at each location 
        at a particular point in time."""

        filename = "std_income_animation.mov"
        if self.check_existance(filename):
            return None

        data = self.get_data_for_animation_std_income()
        self.animate(data,filename)


    def animate_median_income_std_income(self):

        filename = "median_income_std_income.mov"
        if self.check_existance(filename):
            return None

        gb = self.merged_df.groupby(['time','locationID'])
        df = gb.agg({'income':['std','median']}).reset_index()
        
        animation_data = [[] for _ in range(self.num_steps)]

        for index, row in tqdm(df.iterrows()):

            time = int(row['time'])
            mi = row['income']['median']
            std = row['income']['std']

            animation_data[time].append([mi,std])

        self.animate_scatter(
            animation_data,
            filename,
            plot_title='Median Income vs. Standard Deviation Income',
            xlabel='Median Income',
            ylabel='Standard Deviation'
        )


    def plot_location_demographics(self,locationID):

        """plot the number of agents in each income bracket over time
        for a particular location."""

        df = self.merged_df[self.merged_df['locationID']==locationID]
        df = df.groupby(['income_bracket','time']).size().reset_index(name='Count')
        df = pd.pivot_table(df, index='time',columns = 'income_bracket', values='Count')
        
        fig, ax = plt.subplots()
        
        for col in df.columns:
            ax.plot(df[col],label=col)
        
        ax.legend()
        ax.set_title(f'Location {locationID}')
        plt.show()


    def plot_median_incomes(self):

        """Plot the median income time series for each location in the grid."""
        
        filename = 'median_incomes_by_location.png'
        if self.check_existance(filename):
            return None

        df = self.merged_df
        df = df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        df = pd.pivot_table(df,index='time',columns='locationID',values='Median_Income')

        for col in df.columns:
            plt.plot(df[col],color='blue',alpha=0.1)
            plt.title('Median Income for All Locations')
        plt.savefig(os.path.join(self.dirname,filename))


    def plot_income_std(self):

        """Plot the standard deviation of agents' incomes at a particular location over time,
        for each location."""

        filename = 'incomes_std_by_location.png'
        if self.check_existance(filename):
            return None

        # get data
        df = self.merged_df
        df = df.groupby(['time','locationID'])['income'].std().reset_index(name='Income_Std')
        df = pd.pivot_table(df,index='time',columns='locationID',values='Income_Std')
        
        for col in df.columns:
            plt.plot(df[col],color='blue',alpha=0.1)
            plt.title('Income Standard Deviation for All Locations')
        plt.savefig(os.path.join(self.dirname,filename))


    def calculate_move_distances(self):

        """Adds a column to the self.move_df with the distance of each move.
        
        Considers distances that cross the wrap-around bounderies. For example, on a
        10 x 10 grid, locations 9,9 and 0,9 have distance 1 away.
        """

        df = self.move_df

        def calc_distance(row):
            N = self.grid_size

            x1,y1 = row['from_x'], row['from_y']
            x2,y2 = row['to_x'], row['to_y']

            x_dist = min(abs(x1+N-x2),abs(x1-(x2+N)),abs(x1-x2))
            y_dist = min(abs(y1+N-y2),abs(y1-(y2+N)),abs(y1-y2))

            return np.sqrt(x_dist**2 + y_dist**2)

        df['move_distance'] = df.apply(calc_distance,axis=1)


    def plot_move_distance_distribution(self):

        """Generate distribution of all moves, regardless of when they occured."""

        # get move distances
        filename = 'move_distance_distribution.png'
        if self.check_existance(filename):
            return None

        if 'move_distance' not in self.move_df.columns:
            self.calculate_move_distances()
        move_distances = self.move_df['move_distance'].values

        fig = plt.figure()

        bins = np.logspace(np.log10(np.min(move_distances+0.01)), 
                       np.log10(np.max(move_distances)), 
                       num=10)
        vals, bins_ = np.histogram(move_distances,bins=bins)

        plt.plot(bins[:-1],vals,'o-')
        plt.title('Move Distance Distribution')
        plt.savefig(os.path.join(self.dirname,filename))


    def plot_move_activity_by_income(self,smoothing=False):

        """Plot the number of moves of agents in each income bracket at each timestep."""

        filename = 'move_activity_by_income_bracket.png'
        if self.check_existance(filename):
            return None

        df = self.merged_move_df
        df = df[df['fromlocID']!=df['tolocID']]
        df = df.groupby(['income_bracket','time']).size().reset_index(name='Count')
        df = pd.pivot_table(df, index='time',columns = 'income_bracket', values='Count')
        
        #plot
        fig, ax = plt.subplots()
        for col in df.columns:
            
            dat = df[col]
            if smoothing:
                window = 1 if self.num_steps<=50 else int(self.num_steps//50)
                dat = dat.rolling(window=window,min_periods=1).mean().values

            ax.plot(dat,label=col)
        ax.legend()
        ax.set_title('Move Activity by Income Bracket')
        plt.savefig(os.path.join(self.dirname,filename))


    def calculate_capacities(self):

        """Load capacity data from self.locations_df"""

        N = self.grid_size
        capacities = np.empty((N,N))

        for index, row in self.locations_df.iterrows():
            x,y = row['coords']
            capacities[y][x] = row['capacity']

        return capacities


    def plot_capacities(self):
        
        """Visualize capacities on grid."""

        filename = 'capacities.png'
        if self.check_existance(filename):
            return None


        fig,ax  = plt.subplots()

        im = ax.imshow(self.calculate_capacities())
        fig.colorbar(im,ax=ax)
        plt.savefig(os.path.join(self.dirname,filename))


    def calculate_flows(self):

        """Calculate the aggregated number of moves at each location over the course of the simulation."""

        location2coords = {k:v for k,v in zip(self.locations_df['locationID'],self.locations_df['coords'])}

        df = self.move_df
        df_outflow = df.groupby(['fromlocID','time']).size().reset_index(name='Count')
        #df_outflow = df.pivot_table(index='time',columns='fromlocationID',values='Count')
        plt.hist(df_outflow['Count'].values)
        plt.show()

        df_inflow = df.groupby(['tolocID','time']).size().reset_index(name='Count')
        plt.hist(df_inflow['Count'].values)
        plt.show()
        #df_inflow = df.pivot_table(index='time',columns='tolocationID',values='Count')

        N = self.grid_size
        inflows = np.zeros((self.num_steps,N,N))
        outflows = np.zeros((self.num_steps,N,N))

        for index, row in tqdm(df_outflow.iterrows()):
            t = row['time']
            loc = row['fromlocID']
            x,y = location2coords[loc]
            outflows[t][y][x] = row['Count']

        for index, row in tqdm(df_inflow.iterrows()):
            t = row['time']
            loc = row['tolocID']
            x,y = location2coords[loc]
            outflows[t][y][x] = row['Count']

        total_flows = inflows+outflows

        return total_flows, inflows, outflows


    def animate_flows(self):

        """The number of agents moving to and from a location at each timestep.
        
        Not a super useful visualization because moves are infrequent and sparse."""

        filename = 'animate_flows.mov'
        if self.check_existance(filename):
            return None

        total_flows, inflows, outflows = self.calculate_flows()
        data_array = np.array([[inflows, outflows, total_flows]])

        self.multi_animate(
            data_array,
            filename,
            plot_titles=np.array([['Inflows','Outflows','Total Flows']]),
            figsize=(10,4)
        )


    def plot_flows(self):

        """plot aggregated flows (see `calculate_flows()`) and capacities for reference."""

        filename = 'flows.png'
        if self.check_existance(filename):
            return None

        df = self.move_df
        df_outflow = df.groupby(['fromlocID']).size().reset_index(name='Count')
        df_inflow = df.groupby(['tolocID']).size().reset_index(name='Count')

        N = self.grid_size
        inflows = np.zeros((N,N))
        outflows = np.zeros((N,N))

        location2coords = {k:v for k,v in zip(self.locations_df['locationID'],self.locations_df['coords'])}

        for index, row in tqdm(df_outflow.iterrows()):
            loc = row['fromlocID']
            x,y = location2coords[loc]
            outflows[y][x] = row['Count']

        for index, row in tqdm(df_inflow.iterrows()):
            loc = row['tolocID']
            x,y = location2coords[loc]
            outflows[y][x] = row['Count']

        total_flows = inflows+outflows

        fig, ax = plt.subplots(1,3,figsize=(10,4))

        # flows
        im0 = ax[0].imshow(total_flows,norm=colors.LogNorm())
        ax[0].set_title('Total Flows')
        fig.colorbar(im0,ax=ax[0],shrink = 0.5)

        # normalized flows
        im1 = ax[1].imshow(total_flows/self.calculate_capacities(),norm=colors.LogNorm())
        ax[1].set_title('Normalized Flows')
        fig.colorbar(im1,ax=ax[1],shrink=0.5)

        im2 = ax[2].imshow(self.calculate_capacities())
        ax[2].set_title('Location Capacities')
        fig.colorbar(im1,ax=ax[2],shrink = 0.5)

        for a in ax.flatten():
            a.set_axis_off()

        plt.savefig(os.path.join(self.dirname,filename))


    def plot_capacity_distribution(self):

        """See the distribution of all location capacities. Expecting a power law."""

        filename = 'capacity_distribution.png'
        if self.check_existance(filename):
            return None

        caps = self.calculate_capacities()

        out,bins = np.histogram(caps,bins=np.logspace(np.log10(1),np.log10(100), 20),density=True)

        fig = plt.figure()
        plt.title('Distribution of Location Capacities')
        plt.plot(bins[:-1],out,'o-')
        plt.yscale('log')
        plt.savefig(os.path.join(self.dirname,filename))


    def plot_mean_location_score(self, smoothing = True):

        """Each time an agent decides to move, they score locations
        and pick the best one. All those scores for all agents moving at each
        time step are aggregated."""

        filepath = os.path.join(self.dirname,'mean_location_score.png')
        if self.check_existance(filepath):
            return None

        gb = self.possible_location_score_df.groupby('time',as_index=False)

        aggregations = {
            'total_score': 'mean',
            'score_pop_dens': 'mean',
            'score_job_opp':'mean',
            'score_median_income':'mean',
            'score_housing_cost':'mean'
        }

        df = gb.agg(aggregations)

        fig, ax = plt.subplots()

        for col in df.columns:
            if col=='time':
                continue
            vals = df[col].rolling(window=1,min_periods=1).mean()
            ax.plot(vals,label=col)

        plt.legend()
        plt.title('Mean Location Scores at Each Time Step')
        plt.savefig(filepath)


    def plot_mean_move_decision_score(self):

        """At each time step agents decide where to move. Plot the average of these
        scores over time."""

        filepath = os.path.join(self.dirname,'mean_move_decision_score.png')
        if self.check_existance(filepath):
            return None

        df = self.move_decision_score_df

        fig, ax = plt.subplots()

        for col in df.columns:
            if col=='time':
                continue
            vals = df[col].rolling(window=1,min_periods=1).mean()
            ax.plot(vals,label=col)

        plt.legend()
        plt.savefig(filepath)


    def animate_median_income_source_destination(self):

        filename = 'median_income_source_destination.mov'
        if self.check_existance(filename):
            return None

        df = self.merged_df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        median_income_df = pd.pivot_table(df,index='time',columns='locationID',values='Median_Income')

        animation_data = [[] for _ in range(self.num_steps)]

        for index, row in self.move_df.iterrows():

            fromlocID = row['fromlocID']
            tolocID = row['tolocID']

            if fromlocID == tolocID:
                continue

            time = int(row['time'])

            median_income_from = median_income_df[fromlocID][time]
            median_income_to = median_income_df[tolocID][time]

            animation_data[time].append([median_income_from,median_income_to])

        self.animate_scatter(
            animation_data,
            filename,
            plot_title='Median Income of Source and Destination',
            xlabel='Median Income of From Location',
            ylabel='Median Income of To Location')


    def plot_average_median_income_vs_standard_deviation_median_income(self):

        """Look at how much median income changes over time, vs what typical median income
        for a location is."""

        filename = 'average_median_income_vs_std_median_income.png'
        if self.check_existance(filename):
            return None

        df = self.merged_df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        df = pd.pivot_table(df,index='time',columns='locationID',values='Median_Income')

        means, stds = [],[]
        for locID in df.columns:

            means.append(df[locID].mean())
            stds.append(df[locID].std())

        fig = plt.figure()
        plt.scatter(means,stds)
        plt.title('Mean vs. Standard Deviation of Median Income Time Series')
        plt.xlabel('Mean of Median Income Time Series')
        plt.ylabel('Standard Deviation of Median Income Time Series')
        plt.savefig(os.path.join(self.dirname,filename))


    def plot_agents_income_distribution(self):

        filename = 'income_distribution.png'
        if self.check_existance(filename):
            return None

        fig = plt.figure()
        plt.hist(self.agents_df['income'],bins=20,density=True)
        plt.title('Distribution of Agent Incomes')
        plt.savefig(os.path.join(self.dirname,filename))
        

    def animate_median_income_capacity(self):

        filename = 'median_income_vs_capacity.mov'
        if self.check_existance(filename):
            return None

        left = self.merged_df.groupby(['time','locationID'])['income'].median().reset_index(name='Median_Income')
        df = pd.merge(left,self.locations_df,on='locationID')
        
        animation_data = [[] for _ in range(self.num_steps)]

        for index, row in df.iterrows():

            time = row['time']
            mi = row['Median_Income']
            cap = row['capacity']

            animation_data[time].append([mi,cap])

        self.animate_scatter(
            animation_data,
            filename,
            plot_title='Median Income vs. Capacity of Locations',
            xlabel='Median Income',
            ylabel='Capacity'
        )


    def plot_location_score_by_income(self):

        filename = 'location_scores_by_income.png'
        if self.check_existance(filename):
            return None

        df = self.location_score_df
        df = pd.merge(df,self.agents_df,on='agentID')
        df = df.groupby(['time','income_bracket']).agg({'total_score':'mean'}).reset_index()
        df = pd.pivot_table(df,index = 'time',columns='income_bracket',values='total_score')

        fig, ax = plt.subplots()

        for ib in df.columns:
            dat = df[ib].rolling(window=40,min_periods=1).mean().values
            ax.plot(dat,label=ib)

        ax.legend()
        plt.title('Average Location Score by Income Bracket')
        plt.savefig(os.path.join(self.dirname,filename))


    def plot_mean_pre_post_covid(self):

        filename = 'median_income_pre_post_covid.png'
        if self.check_existance(filename):
            pass #return None

        if self.covid_intervention_time > self.num_steps:
            print("Not applicable")
            return None

        df = self.merged_df.groupby(['time','locationID'])['income'].median().reset_index(name='median_income')
        df = pd.merge(df,self.locations_df,on='locationID')
        df = pd.pivot_table(df,index='time',columns='locationID',values='median_income')
        
        # get location properties
        loc_df = self.locations_df

        quartile_labels = ['0-25%','25-50%','50-75%','75-100%']
        loc_df['capacity_quartile'] = pd.qcut(loc_df['capacity'],q=4,labels=quartile_labels)
        
        locID2capacity = {idx:cap for idx, cap in zip(loc_df['locationID'].values,loc_df['capacity'])}
        locID2capacityquartile = {idx:cap for idx, cap in zip(loc_df['locationID'].values,loc_df['capacity_quartile'])}

        # data storage
        mean_of_medians_pre = []
        mean_of_medians_post = []
        caps = []
        capquartile2diff = {k:[] for k in quartile_labels}

        cov = self.covid_intervention_time

        for locID in df.columns:
            vals = df[locID].values
            
            pre = np.mean(vals[:cov])
            post = np.mean(vals[cov:])

            mean_of_medians_pre.append(pre)
            mean_of_medians_post.append(post)
            caps.append(np.sqrt(locID2capacity[locID]))
            capquartile2diff[locID2capacityquartile[locID]].append(post-pre)

        caps = np.array(caps)

        #### fig 1 ####

        fig, ax = plt.subplots(figsize=(7,7))
        s = ax.scatter(mean_of_medians_pre,mean_of_medians_post,alpha=0.4,s=10*caps,c=caps)
        cbar = fig.colorbar(s,ax=ax,shrink=0.7)
        cbar.set_label('Location Capacity')

        ax.set_xlabel('Median Income (pre-COVID)')
        ax.set_ylabel('Median Income (post-COVID)')
        ax.set_xlim(0,45000)
        ax.set_ylim(0,45000)
        ax.set_aspect('equal')
        ax.plot([0,45000],[0,45000],'--',color='black')

        plt.savefig(os.path.join(self.dirname,filename))

        #### fig 2 ####
        filename2 = 'diff_median_income_by_capacity.png'

        fig, ax = plt.subplots()

        ps = [kdeplot(x,label = k) for k,x in capquartile2diff.items()]
        plt.axvline(0,linestyle='--',color='black')
        
        plt.legend()
        plt.savefig(os.path.join(self.dirname,filename2)) 





    def plot_capacity_preference_vs_availability(self):

        filename='capacity_preference_availability.png'
        if self.check_existance(filename):
            return None

        caps = self.locations_df['capacity']
        min_cap, max_cap = min(caps),max(caps)
        loc_counts,bins = np.histogram(caps,bins=np.logspace(np.log10(min_cap),np.log10(max_cap), 30))

        spots_available = loc_counts*bins[1:]

        preferences = self.agents_df['pref_pop_density']

        pref_counts,bins = np.histogram(preferences,bins=bins)

        fig = plt.figure()

        plt.plot(spots_available,'o-',label='Spots Available')
        plt.plot(pref_counts,'o-',label='Number of Agents')
        plt.yscale('log')
        plt.legend()
        plt.show()
        
        
if __name__ == "__main__":
    
    foldername = 'trial_0'
    experiment_name = 'analayze5'

    if not foldername:
        folders = [f for f in os.listdir('data/') if not f.startswith('.')]
        foldername = max(folders,key=lambda f: datetime.strptime(f,"%m-%d_%H-%M-%S"))
        print("Analyzing folder {}".format(foldername))

    sa = SimulationAnalysis(experiment_name, foldername)
    print("loaded")
    # sa.plot_flows()
    # time.sleep(1)
    # sa.plot_capacity_distribution()
    # time.sleep(1)
    # sa.plot_capacities()
    # sa.plot_move_distance_distribution()
    # #sa.animate_population_density()
    # sa.animate_population_density_by_salary()
    # sa.plot_move_activity_by_income(smoothing=True)
    # #sa.plot_location_demographics(0)
    # sa.plot_median_incomes()
    # sa.plot_income_std()

    # sa.plot_mean_location_score()
    # sa.plot_mean_move_decision_score()

    # sa.animate_median_income()
    # sa.animate_std_income()
    # sa.animate_median_income_source_destination()
    # sa.plot_average_median_income_vs_standard_deviation_median_income()
    # sa.plot_agents_income_distribution()

    # sa.animate_median_income_capacity()

    sa.plot_location_score_by_income()

    # sa.animate_median_income_std_income()

    sa.plot_mean_pre_post_covid()

    #sa.plot_capacity_preference_vs_availability()