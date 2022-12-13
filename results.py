import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from parameters import *
from tqdm import tqdm
from gen_random_data import RandomData
import sys
from leave_out import *


def optimal_vs_oos_simple_plot(exp_type):
    models = experiments[exp_type]
    ax_legend = []
    [ax_legend.append(f'c = {np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2)} OOS') for model in
     models]
    [ax_legend.append(f'c = {np.round(model.par.simulated_data.c / model.par.plo.train_frac, 2)} Optimal') for model
     in models]
    shrinkage_list = models[0].par.plo.shrinkage_list
    ones = np.ones(models[0].par.plo.shrinkage_list.shape)
    colors = []
    [colors.append(f'C{i}') for i in range(len(models))]

    names = ['sharpe', 'mse']
    for name in names:
        [plt.plot(shrinkage_list, models[i].oos_perf_est[name], color=colors[i])
         for i in range(len(models))]

        if name == 'sharpe':
            [plt.plot(shrinkage_list, ones * models[i].oos_optimal_sharpe
                      , color=colors[i], linestyle='dashed')
             for i in range(len(models))]

        if name == 'mse':
            [plt.plot(shrinkage_list, ones * models[i].oos_optimal_mse
                      , color=colors[i], linestyle='dashed')
             for i in range(len(models))]


        plt.title(exp_type)
        plt.legend(ax_legend, loc='upper right')
        plt.xlabel('z')
        plt.ylabel(name)
        plt.savefig('plots/'+name+exp_type.replace(" ", "")+'.jpg')
        plt.close()



if __name__ == "__main__":

    folder_to_check = 'data_style'

    par_list = []
    loo_list = []
    experiment_title_list = []
    for f in os.listdir(f'res/{folder_to_check}/'):
        if 'leave_out.p' in os.listdir(f'res/{folder_to_check}/{f}/'):
            par = Params()
            par.load(f'res/{folder_to_check}/{f}/')
            loo = LeaveOut()
            loo.load(f'res/{folder_to_check}/{f}/')
            loo.par = par
            loo_list.append(loo)
            loo.par.update_model_name()
            experiment_title_list.append(loo.par.experiment_title)

    experiment_title_list = list(set(experiment_title_list))

    experiments = {}

    for title in experiment_title_list:
        experiments[title] = []

    for loo in loo_list:
        experiments[loo.par.experiment_title].append(loo)

    os.makedirs('plots', exist_ok=True)
    [optimal_vs_oos_simple_plot(exp_type) for exp_type in experiments]






