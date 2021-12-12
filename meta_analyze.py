import os
import pandas as pd
import json
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import cm
import matplotlib.pyplot as plt

from analyze import SimulationAnalysis

class ExperimentAnalysis:

    def __init__(self,experiment_name):

        self.dirname = os.path.join('data',experiment_name)
        self.experiment_name = experiment_name

        with open(os.path.join(self.dirname,'trials.jsonl')) as f:
            self.trials_df = pd.DataFrame([json.loads(x) for x in f.readlines()])

        self.trial_sims = [SimulationAnalysis(experiment_name,f'trial_{i}') for i in range(len(self.trials_df))]


    def compare_loc_score_distributions(self,param):

        list_of_loc_score_diffs = []
        param_vals = []
        for sa in self.trial_sims:
            s = sa.get_difference_location_score()
            param_vals.append(getattr(sa,param))
            list_of_loc_score_diffs.append(s)

        all_score_diffs = np.array(list_of_loc_score_diffs).flatten()
        print(all_score_diffs.shape)
        min_diff = np.min(all_score_diffs)
        max_diff = np.max(all_score_diffs)

        #print(min_diff,max_diff)

        x = np.linspace(min_diff,max_diff,500)

        kernels = [gaussian_kde(s) for s in list_of_loc_score_diffs]
        ys = [k(x) for k in kernels]

        fig, ax = plt.subplots()

        #viridis = cm.get_cmap('viridis',10)

        for y,p in zip(ys,param_vals):
            plt.plot(x,y,label=p)

        plt.legend()
        plt.show()

if __name__ == "__main__":

    ea = ExperimentAnalysis('update_loc_score_data')

    ea.compare_loc_score_distributions('income_distribution')