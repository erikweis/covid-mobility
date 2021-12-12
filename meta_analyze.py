import os
import pandas as pd
import json

from analyze import SimulationAnalysis

class ExperimentAnalysis:

    def __init__(self,experiment_name):

        self.dirname = os.path.join('data',experiment_name)
        self.experiment_name = experiment_name

        with open(os.path.join(self.dirname,'trials.jsonl')) as f:
            self.trials_df = pd.DataFrame([json.loads(x) for x in f.readlines()])

        self.sims = [SimulationAnalysis(experiment_name,f'trial_{i}') for i in range(len(self.trials_df))]

if __name__ == "__main__":

    ea = ExperimentAnalysis('powerlaw_income')