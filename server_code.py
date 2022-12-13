import os
import pandas as pd
import numpy as np
from parameters import *
from tqdm import tqdm  # use for showing progress in loops
from gen_random_data import RandomData
import sys
from leave_out import *
import pickle


def run_experiment(gl, grid_id):
    par.update_param_grid(gl, grid_id)
    par.update_model_name()

    save_folder = f'res/{par.name}/'
    os.makedirs(save_folder, exist_ok=True)

    data = RandomData(par)

    labels, features = data.gen_random_data(create=True)
    loo = LeaveOut(par, labels, features)

    loo.train_test_split()
    loo.train_model()
    loo.ins_performance()
    loo.oos_performance()
    loo.save(save_dir=save_folder)
    par.save(save_dir=save_folder)
    print('here', flush=True)


if __name__ == "__main__":
    try:

        grid_id = int(sys.argv[1])
    except:
        print('Debug mode on local machine')
        grid_id = -1

    counter = 0
    max_iter = 1
    gl = [
        ['simulated_data', 'b_star', [0.01,  1]],
        ['simulated_data', 'alpha', [0.1, 2]],
        ['simulated_data', 'beta_and_psi_link', [0.1, 2]],
        ['simulated_data', 't', [1000, 5000]],
        ['simulated_data', 'c', [0.2, 1, 5]]
    ]

    for g in gl:
        max_iter = max_iter * len(g[2])

    par = Params()

    # creates folder with specific experiment result type
    par.name_detail = 'data_style'

    # update grid

    if grid_id < 0:
        while counter < max_iter:
            run_experiment(gl, counter)
            counter += 1
    else:
        run_experiment(gl, grid_id)
