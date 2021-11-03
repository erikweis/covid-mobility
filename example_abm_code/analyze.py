from networkx.algorithms import similarity
import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import os
import networkx as nx
from idea import Idea
from itertools import combinations
import json
import pickle
import plotly.graph_objects as go
from tqdm import tqdm
import tnetwork as tn
import tnetwork.DCD as DCD

class SimulationAnalysis:

    def __init__(self, folder_name):

        self.dirname = 'data/'+folder_name
        
        #data frames from files
        self.idea_pop_data = pd.read_csv(self.dirname+'/idea_popularity.csv')

        self.idea_data = pd.read_csv(self.dirname+'/ideas.csv',dtype={'ideaID':int,'bitstring':str})
        self.beliefs_data = pd.read_csv(self.dirname+'/beliefs.csv')

        with open(self.dirname+'/run_params.json','r') as f:
            params =json.load(f)

        for attr,value in params.items():
            self.__setattr__(attr,value)

    def plot_idea_popularity(self):

        fig, ax = plt.subplots()

        #idea popularity
        ideaidx2smoothedpopularity = self.get_ideaidx2smoothedpopularity(window=1)
        ideaidx2totalpopularity = {i:sum(j) for i,j in ideaidx2smoothedpopularity.items()}
        max_total_pop = max(ideaidx2totalpopularity.values())

        for idx, smoothedpop in ideaidx2smoothedpopularity.items():
            x = list(range(len(smoothedpop)))
            ax.plot(x,smoothedpop)

        ax.set_xlim(0,self.simulation_length)
        ax.set_ylim(0,max(max(i) for i in ideaidx2smoothedpopularity.values()))

        plt.show()

    def animate(self):

        fig,ax = plt.subplots(1,2,figsize=(10,5))
        
        #locations
        df = self.loc_data
        df_plot = df[df['time']==0]
        x,y = df_plot['x'].values,df_plot['y'].values
        time_length = max(df['time'].values)+1

        ax[0].set(xlim=(-100,100),ylim=(-100,100))
        scatter=ax[0].scatter(x,y)

        #idea popularity
        ideaidx2smoothedpopularity = self.get_ideaidx2smoothedpopularity()
        ideaidx2totalpopularity = {i:sum(j) for i,j in ideaidx2smoothedpopularity.items()}
        max_total_pop = max(ideaidx2totalpopularity.values())
        ideaidx2linealpha = {idx:t/max_total_pop for idx,t in ideaidx2totalpopularity.items()}

        lines = [ax[1].plot([0],smoothedpop[0],alpha=ideaidx2linealpha[idx])[0] for idx,smoothedpop in ideaidx2smoothedpopularity.items()]

        ax[1].set_xlim(0,time_length)
        ax[1].set_ylim(0,max(max(i) for i in ideaidx2smoothedpopularity.values()))

        def update(frame_number):
            df_plot = df[df['time']==frame_number]

            x,y = df_plot['x'].values, df_plot['y'].values

            scatter.set_offsets(list(zip(x,y)))

            for smoothedpop,line in zip(ideaidx2smoothedpopularity.values(),lines):
                #line.set_data(np.arange(frame_number),smooth_time_series(pop_data[i][:frame_number]))
                x_ = np.arange(frame_number)
                y_ = smoothedpop[:frame_number]

                line.set_data(x_,y_)
                
            return scatter, *lines

        anim = animation.FuncAnimation(fig, update,interval=1,frames=np.arange(1,time_length,50),repeat=False)
        plt.show()

    def plot_idea_survival_distribution(self):

        ideaidx2smoothedpopularity = self.get_ideaidx2smoothedpopularity()
        survival_times = [np.sum(np.array(popularity)>0) for popularity in ideaidx2smoothedpopularity.values()]

        plt.scatter(np.arange(len(survival_times)),sorted(survival_times)/max(survival_times))
        plt.xscale('log')
        plt.xlabel('Idea Survival Time')
        plt.ylabel('P(t <= T)')
        plt.title('Cumulative Distribution of Survival Times')
        # vals, bins = np.histogram(survival_times,bins = 40)
        # plt.scatter(bins[:-1],survival_times)
        plt.show()

    def get_idx2smoothedpopularity_at_time(self,time):

        idx2smoothedpopularity = self.get_ideaidx2smoothedpopularity()
        return {idx:pop[time] for idx, pop in idx2smoothedpopularity.items()}


    def get_idx2idea(self):

        if not hasattr(self,'_idx2idea'):
            df = self.idea_data

            ideaIDs = df['ideaID'].values
            ideaBitStrings = df['bitstring'].values

            self._idx2idea = {idx:Idea(idx=idx,bitstring=bitstring) for idx,bitstring in zip(ideaIDs,ideaBitStrings)}

        return self._idx2idea


    def get_idea_network(self,time):

        idx2idea = self.get_idx2idea()
        max_pop = max(max(i) for i in self.get_ideaidx2smoothedpopularity().values())
        idx2popularity = self.get_idx2smoothedpopularity_at_time(time)
        ideaIDs = list(self.get_idx2idea().keys())       

        G = nx.Graph()
        
        #calculate edge weights
        weights = {}
        for i,j in combinations(ideaIDs,2):
            idea_i = idx2idea[i]
            idea_j = idx2idea[j]

            w = idea_i.similarity(idea_j)
            w *= idx2popularity[i]*idx2popularity[j]/(max_pop**2)

            if w>0:
                weights[(i,j)] = w

        edges = [(nodes[0],nodes[1],w) for nodes,w in weights.items()]
        G.add_weighted_edges_from(edges)

        return G


    def plot_idea_network(self,time=1):

        G = self.get_idea_network(time)

        weights = [G[u][v]['weight'] for u,v in G.edges]
        #weights = [w if w>0.6 else 0 for w in weights]
        weights = np.array(weights)*5 #value to scale the weights for visual appeal

        pos = nx.spring_layout(G)

        nx.draw(G,pos=pos,width = weights,alpha=0.2)
        plt.show()

    def animate_idea_networks(self):

        def simple_update(num, pos,G,  ax):
            ax.clear()

            if 100*num//self.simulation_length %10 ==0:
                print(f"{100*num//self.simulation_length}% done")


            G = self.get_idea_network(time=num)

            weights = [G[u][v]['weight'] for u,v in G.edges]
            #weights = [w if w>0.6 else 0 for w in weights]
            weights = np.array(weights)*5 #value to scale the weights for visual appeal

            pos = nx.spring_layout(G,pos=pos,iterations=2)
            nx.draw(G, pos=pos, ax=ax,alpha=0.2,edge_color='gray')

            # Set the title
            ax.set_title("Frame {}".format(num))

        # Build plot
        fig, ax = plt.subplots(figsize=(6,4))

        # Create a graph and layout
        G = self.get_idea_network(time=0)
        pos = nx.spring_layout(G)

        anim = animation.FuncAnimation(fig, simple_update, interval=1, frames=np.arange(1,self.simulation_length,step=20), fargs=(pos, G, ax))
        #ani.save('animation_1.gif', writer='imagemagick')

        print("created animation")

        f = rf"{self.dirname}/anim.gif"
        writergif = animation.PillowWriter(fps=60) 
        anim.save(f, writer=writergif)

        plt.show()


    def get_ideaidx2smoothedpopularity(self,window=20):
 
        df_ip = self.idea_pop_data.pivot_table(index=['time'],columns=['ideaID'],values='popularity',fill_value=0)
        all_ideaIDs = self.idea_pop_data['ideaID'].unique()

        return {idea_idx:df_ip[idea_idx].rolling(window=window,min_periods=1).mean().values for idea_idx in all_ideaIDs}



    def evolutionary_community_detection(self):

        dg_sn = tn.DynGraphSN(frequency=1)

        for t in range(self.simulation_length):
            
            G = self.get_idea_network(t)
            dg_sn.add_nodes_presence_from(list(G.nodes),[t])
            dg_sn.add_interactions_from(list(edge for edge in G.edges),[t])

        com = DCD.iterative_match(dg_sn)
        communities = com.communities_sn_by_sn()

        print(list(communities.keys()))

        plot = tn.plot_longitudinal(dg_sn,com,height=1000)
        plt.show()



    def get_community_strength(self,community,G):
        
        S=G.subgraph(list(community))
        internal_edges=S.number_of_edges()
        degrees=[G.degree(node) for node in S.nodes]
        external_edges=sum(degrees)-2*internal_edges
        if external_edges<1:
            return 1
        return internal_edges/external_edges


    def draw_community_structure(self):
        
        partition_path= os.path.join(self.dirname,'partition_list.pickle')
        
        #get all graphs
        times = list(range(0,self.simulation_length,self.simulation_length//15))
        G_list = [self.get_idea_network(t) for t in times]

        #if paritions already calculated, load them, otherwise, calculate them
        if os.path.isfile(partition_path):
            filehandler=open(partition_path,'rb')
            self.partitions=pickle.load(filehandler)
            filehandler.close()
        else:

            self.partitions=[]
            for G in tqdm(G_list):
                partitions_list=nx.community.girvan_newman(G)
                keep=max(partitions_list,key=lambda p:nx.community.quality.modularity(G,p))
                self.partitions.append(keep)
            
            with open(partition_path,'wb') as f:
                pickle.dump(self.partitions,f)

        
        #edge list for visualization
        edge_list=[]
        
        for i in range(len(self.partitions)-1):
            for j in range(len(self.partitions[i])):
                this_community=self.partitions[i][j]
                for k in range(len(self.partitions[i+1])):
                    next_community=self.partitions[i+1][k]
                    weight=len(this_community.intersection(next_community))
                    if weight>0:
                        edge_list.append(((i,j),(i+1,k),weight))
        
        
        #create visual
        nodes=list(set([edge[0] for edge in edge_list]+[edge[1] for edge in edge_list]))
        mapping={node:i for i,node in enumerate(nodes)}
        label=[node[1] for node in nodes]
        x=[node[0] for node in nodes]
        strength=[]
        for node in nodes:
            commune=self.partitions[node[0]][node[1]]
            t=times[node[0]]
            G=G_list[node[0]]
            strength.append(self.get_community_strength(commune,G))
        label=[round(s,1) for s in strength]
        strength=[x/max(strength) for x in strength]
        strength=[(x+0.5)/1.5 for x in strength]
        
        
        source=[]
        target=[]
        value=[]
        for index, edge in enumerate(edge_list):
            source.append(edge[0])
            target.append(edge[1])
            value.append(edge[2])
        
        
        source=[mapping[s] for s in source]
        target=[mapping[t] for t in target]
        
        fig = go.Figure(go.Sankey(
            arrangement = "snap",
            node = {
                "label": label,
                "x": x,
                'pad':10},  # 10 Pixels
            link = {
                "source": source,
                "target": target,
                "value": value}))
        
        plt.show()
        return fig

if __name__ == "__main__":
    
    foldername = ''

    if not foldername:
        folders = [f for f in os.listdir('data/') if not f.startswith('.')]
        foldername = max(folders,key=lambda f: datetime.strptime(f,"%m-%d_%H-%M-%S"))
        print("Analyzing folder {}".format(foldername))

    sa = SimulationAnalysis(foldername)
    print("loaded")
    #sa.plot_idea_popularity()
    sa.plot_idea_survival_distribution()