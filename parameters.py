import itertools
from enum import Enum
import numpy as np
import pandas as pd
import socket

# parameters
# Created in the name of Antoine Didisheim, by his disiple at 6.12.22
# job: store default parameters used throughout the projects in single .py



##################
# Constant
##################

class DataUsed (Enum):
    INS = 1
    OOS = 2
    TOTAL = 3

class Estimators (Enum):
    MEAN = 1
    STD = 2
    PI = 3
    PI_2 = 4
    R_2 = 5
    MSE = 6

class Constant:
    MAX_OPT = 1000
    GRID_SIZE = 1000
    BASE_DIR = './'




##################
# params classes
##################

class ParamsLeaveOut:
    def __init__(self):
        self.train_frac = 0.5
        self.shrinkage_list = np.linspace(0.1, 10, 100)



class SimulatedDataParams:
    def __init__(self):
        self.alpha = 1
        self.b_star = 0.01
        self.seed = 0
        self.beta_and_psi_link = 2
        self.noise_size = 0
        self.activation = 'linear'
        self.number_neurons = 1
        self.t = 100
        self.c = 2
        self.p = int(self.c * self.t)
        self.simple_beta = False

    def get_name(self):
        n = f'T{self.t}C{str(self.c)}b_star{self.b_star}l{self.beta_and_psi_link }alpha{self.alpha}'
        return n



# store all parameters into a single object
class Params:
    def __init__(self):
        self.name_detail = 'default'
        self.name = ''
        self.seed = 12345
        self.plo = ParamsLeaveOut()
        self.simulated_data = SimulatedDataParams()
        self.process = None
        self.update_model_name()
        self.experiment_title = None

    def update_model_name(self):
        n = self.name_detail
        n += f'/b_star{self.simulated_data.b_star}beta_and_psi_link{self.simulated_data.beta_and_psi_link}' \
             f'alpha{self.simulated_data.alpha}c{self.simulated_data.c}t{self.simulated_data.t}'
        self.experiment_title = f'T={int(self.plo.train_frac * self.simulated_data.t) }, T_1={int((1-self.plo.train_frac) * self.simulated_data.t) }  ' \
                                f'\n b_*={self.simulated_data.b_star} & l={self.simulated_data.beta_and_psi_link} & a={self.simulated_data.alpha} '
        self.name = n

    def print_values(self):
        """
        Print all parameters used in the model
        """
        for key, v in self.__dict__.items():
            try:
                print('########', key, '########')
                for key2, vv in v.__dict__.items():
                    print(key2, ':', vv)
            except:
                print(v)

    def update_param_grid(self, grid_list, id_comb):
        ind = []
        for l in grid_list:
            t = np.arange(0, len(l[2]))
            ind.append(t.tolist())
        combs = list(itertools.product(*ind))
        print('comb', str(id_comb + 1), '/', str(len(combs)))
        c = combs[id_comb]

        for i, l in enumerate(grid_list):
            self.__dict__[l[0]].__dict__[l[1]] = l[2][c[i]]

    def save(self, save_dir, file_name='/parameters.p'):
        # simple save function that allows loading of deprecated parameters object
        df = pd.DataFrame(columns=['key', 'value'])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                    df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                df = df.append(temp)
        df.to_pickle(save_dir + file_name, protocol=4)

    def load(self, load_dir, file_name='/parameters.p'):
        # simple load function that allows loading of deprecated parameters object
        df = pd.read_pickle(load_dir + file_name)
        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=['key', 'value'])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                        df = df.append(temp)

                except:
                    temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df['key'] == str(key) + '_' + str(key2), 'value']
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            print('#### Loaded parameters object is depreceated, default version will be used')
                        print('Parameter', str(key) + '.' + str(key2), 'not found, using default: ',
                              self.__dict__[key].__dict__[key2])

            except:
                t = df.loc[df['key'] == str(key), 'value']
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                        print('#### Loaded parameters object is depreceated, default version will be used')
                    print('Parameter', str(key), 'not found, using default: ', self.__dict__[key])




