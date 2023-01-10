import os
import pandas as pd
import numpy as np
from parameters import *
from tqdm import tqdm  # use for showing progress in loops
from gen_random_data import RandomData
import sys
from leave_out import *
from data import *
import pickle
import random
import seaborn as sns


def gaussian_kernel(X: np.ndarray,
                    X_test: np.ndarray = None,
                    norm_band_width: int = 1):
    if X_test is None:
        X_test = X

    norm_x = (X ** 2).sum(1)
    norm_x_test = (X_test ** 2).sum(1)
    inner_prod = X_test @ X.T
    tmp = norm_x_test.reshape(-1, 1) + norm_x.reshape(1, -1) - 2 * inner_prod
    band_width = norm_band_width * X.shape[1]
    kernel = np.exp(-tmp / band_width) / X.shape[0]

    return kernel


def train_kernel_model(eigval, eigvec, labels, shrinkage_list):
    inv_eigvals = [1 / (eigval.reshape(-1, 1) + z) for z in shrinkage_list]
    beta_kernel = [eigvec @ (inv_eigval * eigvec.T) @ labels for inv_eigval in inv_eigvals]

    return beta_kernel


def kare(eigval, eigvec, labels, shrinkage_list):

    inv_eigvals = [1 / (eigval.reshape(-1, 1) + z) for z in shrinkage_list]
    inv_eigvals_squared = [inv_eigval ** 2 for inv_eigval in inv_eigvals]
    denominator = [(np.sum(inv_eigval) / len(labels)) ** 2 for inv_eigval in inv_eigvals]
    numerator = [labels.T @ eigvec @ (e * eigvec.T) @ labels / len(labels) for e in inv_eigvals_squared]
    kare = [(numerator[i] / denominator[i])[0] for i in range(len(shrinkage_list))]

    return kare


def calculate_mse(predictions, true_labels):

    mse = [np.mean((pred - true_labels) ** 2) for pred in predictions]

    return mse


def gen_random_features_from_random_sub_sample(raw_features_oos:np.ndarray,
                                               raw_features_ins:np.ndarray,
                                               par:Params,
                                               norm_band_width:int =1):
    band_width = norm_band_width * raw_features_oos.shape[1]

    specification = {'distribution': 'normal',
                     'distribution_parameters': [0, 2/band_width],
                     'activation': 'cos_and_sin',
                     'number_features': int(raw_features_ins.shape[0] * par.data.c),
                     'bias_distribution': 'uniform',
                     'bias_distribution_parameters': [0,2*np.pi]}

    features_oos = RandomFeaturesGenerator.generate_random_neuron_features(
        raw_features_oos,
        par.data.seed,
        **specification
    )
    features_ins = RandomFeaturesGenerator.generate_random_neuron_features(
        raw_features_ins,
        par.data.seed,
        **specification
    )

    return np.sqrt(2)*features_oos, np.sqrt(2)*features_ins


def run_experiment(gl,
                   grid_id,
                   par):
    par.update_param_grid(gl, grid_id)
    par.update_mnist_basic_title()
    # pushes predictions to 1 and -1
    par.plo.classification = True

    save_folder = f'res/{par.name}/'
    os.makedirs(save_folder, exist_ok=True)

    data = Data(par)
    labels_oos, labels_ins, raw_features_oos, raw_features_ins, features_oos, features_ins = data.load()
    random.seed(par.data.seed)
    train_chosen_indices = random.sample(range(0, labels_ins.shape[0]), par.data.train_sub_sample)
    test_chosen_indices = random.sample(range(0, labels_oos.shape[0]), par.data.test_sub_sample)

    labels_ins = labels_ins[train_chosen_indices]
    labels_oos = labels_oos[test_chosen_indices]

    raw_features_ins = raw_features_ins[train_chosen_indices, :]
    raw_features_oos = raw_features_oos[test_chosen_indices, :]

    features_oos, features_ins = gen_random_features_from_random_sub_sample(raw_features_oos,
                                                                            raw_features_ins,
                                                                            par)

    loo = LeaveOut(par)
    loo.add_splitted_data(labels_oos, labels_ins, features_oos, features_ins)
    loo.train_model()
    loo.ins_performance()
    loo.oos_performance()

    # save what you need
    loo.save(save_dir=save_folder)
    par.save(save_dir=save_folder)
    print('Model trained', flush=True)


if __name__ == "__main__":
    try:

        grid_id = int(sys.argv[1])
    except:
        print('Debug mode on local machine')
        grid_id = 0

    counter = 0
    max_iter = 1

    # c_list = np.linspace(0.1, 5, 15)
    # c_list = [0.1,0.25,0.5,1,2,5]
    # c_list = [5]
    # fix distinction between random and simple features
    gl = [
        ['data', 'c', [0.1, 1, 5]],
        ['data', 'activation', ["cos_and_sin", 'tanh']]
    ]

    for g in gl:
        max_iter = max_iter * len(g[2])

    par = Params()
    # creates folder with specific experiment result type
    par.name_detail = 'mnist_first_test'
    par.plo.shrinkage_list = np.logspace(-8,2,20)

    data = Data(par)
    labels_oos, labels_ins, raw_features_oos, raw_features_ins, features_oos, features_ins = data.load()
    random.seed(par.data.seed)
    train_chosen_indices = random.sample(range(0, labels_ins.shape[0]), par.data.train_sub_sample)
    test_chosen_indices = random.sample(range(0, labels_oos.shape[0]), par.data.test_sub_sample)

    labels_ins = labels_ins[train_chosen_indices]
    labels_oos = labels_oos[test_chosen_indices]

    raw_features_ins = raw_features_ins[train_chosen_indices, :]
    raw_features_oos = raw_features_oos[test_chosen_indices, :]

    norm_l = 1

    kernel_oos = gaussian_kernel(X=raw_features_ins,
                                 X_test=raw_features_oos,
                                 norm_band_width=norm_l)
    gram_matrix = gaussian_kernel(X=raw_features_ins,
                                  norm_band_width=norm_l)
    eigval, eigvec = np.linalg.eigh(gram_matrix)

    beta_kernel = train_kernel_model(eigval=eigval, eigvec=eigvec, labels=labels_ins,
                                     shrinkage_list=par.plo.shrinkage_list)

    predictions_oos = [kernel_oos @ beta for beta in beta_kernel]
    predictions_ins = [gram_matrix @ beta for beta in beta_kernel]

    risk = calculate_mse(predictions=predictions_oos, true_labels=labels_oos)
    # train_err = calculate_mse(predictions=predictions_ins, true_labels=labels_ins)
    # kare_est = kare(eigval=eigval,
    #                 eigvec=eigvec,
    #                 labels=labels_ins,
    #                 shrinkage_list=par.plo.shrinkage_list)
    # plt.plot(par.plo.shrinkage_list, kare_est)
    # plt.plot(par.plo.shrinkage_list, risk)
    # plt.plot(par.plo.shrinkage_list, train_err)
    # legend = ['KARE', 'Risk', 'Train err']
    # plt.xlabel('z')
    # plt.ylabel('MSE')
    # plt.xscale('log')
    # plt.legend(legend)
    # plt.plot()
    # plt.show()

    par.data.c = 10

    features_oos, features_ins = gen_random_features_from_random_sub_sample(raw_features_oos,
                                                                            raw_features_ins,
                                                                            par)

    par.plo.g = 10**(-3)
    loo = LeaveOut(par)
    loo.add_splitted_data(labels_oos, labels_ins, features_oos, features_ins)
    loo.train_model()
    loo.ins_performance()
    loo.oos_performance()


    plt.plot(par.plo.shrinkage_list, loo.oos_perf_est['mse'])
    plt.plot(par.plo.shrinkage_list, risk)
    # plt.plot(par.plo.shrinkage_list, np.ones(par.plo.shrinkage_list.shape)*loo.oos_optimal_mse)
    # plt.plot(par.plo.shrinkage_list, np.ones(par.plo.shrinkage_list.shape)*loo.infeasible_oos_optimal_mse)
    legend = [f'C={par.data.c} RF', 'Kernel']
    plt.xlabel('z')
    plt.ylabel('MSE')
    plt.xscale('log')
    plt.legend(legend)
    plt.plot()
    plt.show()

    samples = np.logspace(1,5,20)
    zz_list = []
    i = 1005
    j = 1008
    norm_band_width = 1
    band_width = norm_band_width * raw_features_oos.shape[1]

    for s in samples:

        specification = {'distribution': 'normal',
                         'distribution_parameters': [0, 2/band_width],
                         'activation': 'cos',
                         'number_features': int(s),
                         'bias_distribution': None,
                         'bias_distribution_parameters': [0]}

        z = RandomFeaturesGenerator.generate_random_neuron_features(
            raw_features_ins[np.array([i,j]),:],
            par.data.seed,
            **specification
        )
        print(z.shape)
        zz_list.append(z[0,:]@z[1,:].T/s)

    goal = np.ones(samples.shape)*gram_matrix[i,j]*raw_features_ins.shape[0]

    plt.plot(samples,zz_list)
    plt.plot(samples,goal)
    plt.legend(['RF','Kernel'])
    plt.xlabel('P')
    plt.xscale('log')
    plt.show()


    #
    # real_gram = gram_matrix*raw_features_ins.shape[0]
    # percentage_difference = 100*np.abs(real_gram - rf_gram)/real_gram
    # sns.heatmap(percentage_difference)
    # plt.show()
    #
    #
    # s = 10**5
    #
    # specification = {'distribution': 'normal',
    #                  'distribution_parameters': [0, 2 / band_width],
    #                  'activation': 'cos',
    #                  'number_features': s,
    #                  'bias_distribution': None,
    #                  'bias_distribution_parameters': [0]}
    #
    # z = RandomFeaturesGenerator.generate_random_neuron_features(
    #     raw_features_ins,
    #     par.data.seed,
    #     **specification
    # )
    #
    # rf_gram = z @ z.T / s
    #
    #
    # percentage_difference = 100*np.abs(real_gram - rf_gram)/real_gram
    # sns.heatmap(percentage_difference)
    # plt.show()








    # plt.plot(par.plo.shrinkage_list, loo.w_mse_ins)
    # plt.plot(par.plo.shrinkage_list, loo.w_mse_oos)
    # legend = ['INS', 'OOS']
    # plt.xlabel('z')
    # plt.ylabel('MSE Weights')
    # plt.xscale('log')
    # plt.legend(legend)
    # plt.plot()
    # plt.show()

    # bandwidth_list = np.logspace(-3, 0, 10)
    # shrinkage = [10 ** (-5)]
    # risk_list = []
    # train_err_list = []
    # kare_list = []
    # for bandwidth in bandwidth_list:
    #     kernel_oos = gaussian_kernel(X=raw_features_ins,
    #                                  X_test=raw_features_oos,
    #                                  norm_band_width=bandwidth)
    #     gram_matrix = gaussian_kernel(X=raw_features_ins,
    #                                   norm_band_width=bandwidth)
    #     eigval, eigvec = np.linalg.eigh(gram_matrix)
    #
    #     beta_kernel = train_kernel_model(eigval=eigval,
    #                                      eigvec=eigvec,
    #                                      labels=labels_ins,
    #                                      shrinkage_list=shrinkage)
    #
    #     predictions_oos = [kernel_oos @ beta for beta in beta_kernel]
    #     predictions_ins = [gram_matrix @ beta for beta in beta_kernel]
    #
    #     risk_list.append(calculate_mse(predictions=predictions_oos, true_labels=labels_oos)[0])
    #     train_err_list.append(calculate_mse(predictions=predictions_ins, true_labels=labels_ins)[0])
    #     kare_list.append(kare(eigval=eigval,
    #                           eigvec=eigvec,
    #                           labels=labels_ins,
    #                           shrinkage_list=par.plo.shrinkage_list)[0])
    #
    # plt.plot(bandwidth_list, kare_list)
    # plt.plot(bandwidth_list, risk_list)
    # plt.plot(bandwidth_list, train_err_list)
    # legend = ['KARE', 'Risk', 'Train err']
    # plt.xlabel('l/d')
    # plt.ylabel('MSE')
    # plt.xscale('log')
    # plt.legend(legend)
    # plt.plot()
    # plt.show()



    # update grid
    #
    # if grid_id < 0:
    #     while counter < max_iter:
    #         run_experiment(gl, counter, par)
    #         counter += 1
    # else:
    #     run_experiment(gl, grid_id, par)
