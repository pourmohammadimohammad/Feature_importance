import numpy
import numpy as np
import pandas as pd
from leaveout import *
from main import *
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.marcenko_pastur import MarcenkoPastur
from parameters import *


# mohammad_is_wrong = RandomFeatures.naive_linear_single_underlying()


# Clean up and test leave two out
# Todo: leave one out and leave two out equivalane
# Todo: Bring everything on server
# Todo: Fix variance estimate

def map_w_to_w_tilde(w_matrix):
    """
    We need to map W_{t_1,t_2} to W_{t_1,t_2}/((1-W_{t1,t1})(1-W_{t2,t2})-W_{t1,t2}^2)
    :param w_matrix:
    :return:
    """
    diag = (1 - np.diag(w_matrix)).reshape(-1, 1) * (1 - np.diag(w_matrix))
    denominator = diag - (w_matrix ** 2)
    w_tilde = w_matrix / denominator
    np.fill_diagonal(w_tilde, 0)
    return w_tilde


def leave_two_out_estimator_vectorized_resolvent(labels: np.ndarray,
                                                 features: np.ndarray,
                                                 eigenvalues: np.ndarray,
                                                 eigenvectors: np.ndarray,
                                                 shrinkage_list: np.ndarray) -> float:
    """
    # Implement leave two out estimators_oos
    # For any matrix A independent of t_1 and t_2
    # E[R_{t_2+1} S_{t_1}'A S_{t_2} R_{t_1+1}]\ =
    \  \beta'\Psi A_left (\hat \Psi + zI)^{-1} A_right \Psi \beta\
    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param eigenvectors:
    :param eigenvalues:
    :param shrinkage_list:
    :return: Unbiased estimator
    """

    W = LeaveOut.smart_w_matrix(features=features,
                                eigenvalues=eigenvalues,
                                eigenvectors=eigenvectors,
                                shrinkage_list=shrinkage_list)

    T = np.shape(features)[0]

    num = (T - 1)  # divided by times to account for W normalization
    labels = labels.reshape(-1, 1)

    estimator_list = [(labels.times @ map_w_to_w_tilde(w_matrix) @ labels / num)[0] for w_matrix in W]

    return estimator_list


def leave_two_out_estimator_vectorized_general(labels: np.ndarray,
                                               features: np.ndarray,
                                               A: np.ndarray) -> float:
    """
    # Implement leave two out estimators_oos
    # For any matrix A independent of t_1 and t_2
    # E[R_{t_2+1} S_{t_1}'A S_{t_2} R_{t_1+1}]\ =
    \  \beta'\Psi A_ right \Psi \beta\
    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param A: Weighting matrix
    :return: Unbiased estimator
    """

    T = np.shape(features)[0]

    num = (T - 1)  # divded by times to account for W normalization

    matrix_multiplied = features.T @ A @ features
    np.fill_diagonal(matrix_multiplied, 0)

    labels = labels.reshape(-1, 1)

    estimator = (labels.times @ matrix_multiplied @ labels / num)[0]

    return estimator


def estimate_mean_leave_two_out(labels: np.ndarray,
                                features: np.ndarray,
                                eigenvalues: np.ndarray,
                                eigenvectors: np.ndarray,
                                shrinkage_list: np.ndarray) -> float:
    [T, P] = features.shape
    c = P / T

    estimator_list = leave_two_out_estimator_vectorized_resolvent(labels,
                                                                  features,
                                                                  eigenvalues,
                                                                  eigenvectors,
                                                                  shrinkage_list)
    m_list = LeaveOut.empirical_stieltjes(eigenvalues, P, shrinkage_list)
    mean_estimator = [(1 - c + c * shrinkage_list[i] * m_list[i]) * estimator_list[i] for i in
                      range(len(shrinkage_list))]

    return mean_estimator


def run_loo(t: int,
            c: float,
            train_frac: float,
            seed: int = None,
            beta_and_psi_link: float = None,
            shrinkage_list: list = None,
            simple_beta: bool = False,
            growing_oos: bool = False) -> object:
    seed = 0 if seed is None else seed

    lo_est = LeaveOut(t, c)
    lo_est.seed = seed

    beta_and_psi_link = lo_est.beta_and_psi_link if beta_and_psi_link is None else beta_and_psi_link
    shrinkage_list = lo_est.shrinkage_list if shrinkage_list is None else shrinkage_list

    lo_est.beta_and_psi_link = beta_and_psi_link
    lo_est.shrinkage_list = shrinkage_list
    lo_est.simulate_date(simple_beta=simple_beta)
    lo_est.train_test_split(train_frac)
    lo_est.train_model()
    estimator_in_sample = lo_est.ins_performance()

    if not growing_oos:
        estimator_out_of_sample = lo_est.oos_performance()
    else:
        estimator_out_of_sample = lo_est.oos_performance_growing_sample()

    return estimator_in_sample, estimator_out_of_sample, lo_est


def plot_sub_plots(estimators, shrinkage_list, ax_legend, c, plot_name):
    times = list(estimators.keys())
    num_seeds = len(estimators[times[0]])
    for name in estimators[times[0]][0].keys():
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
        for i in range(len(times)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][name])
             for j in range(num_seeds)]
            ax[a_0, a_1].set_title(f'T = {times[i]}')

        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
        fig.suptitle(name.upper() + ' ' + plot_name + f" \n  c = {c}", fontsize=12)
        plt.show()


def plot_sub_plots_mean(estimators, shrinkage_list, ax_legend, c, train_frac):
    times = list(estimators.keys())
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):
        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j])
         for j in range(len(estimators[times[i]]))]

        ax[a_0, a_1].set_title(f'T = {times[i]}')

    ax[0, 0].legend(ax_legend, loc='upper right')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    fig.suptitle(' Mean' + f" \n  c = {c} \n train fraction = {train_frac} ", fontsize=12)
    plt.show()


def plot_sub_plots_ins_vs_oos(estimators, estimators_oos, shrinkage_list, ax_legend, c, train_frac, plot_name):
    times = list(estimators.keys())
    num_seeds = len(estimators[times[0]])
    for name in estimators[times[0]][0].keys():
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
        for i in range(len(times)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][name])
             for j in range(num_seeds)]

            [ax[a_0, a_1].plot(shrinkage_list, estimators_oos[times[i]][j][name])
             for j in range(num_seeds)]
            ax[a_0, a_1].set_title(f'T = {times[i]}')

        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
        fig.suptitle(name.upper() + ' ' + plot_name + f" \n  c = {c} \n train fraction = {train_frac} ", fontsize=12)
        plt.show()


def plot_unconditionally(estimators_oos, estimators_ins, true_values, shrinkage_list, c, times,
                         name='mean'):
    avg_perf_OOS = {}
    avg_perf_INS = {}
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):

        avg_perf_OOS[times[i]] = np.zeros(len(estimators_oos[times[i]][0][name]))
        avg_perf_INS[times[i]] = np.zeros(len(estimators_ins[times[i]][0][name]))
        for j in range(len(seeds)):
            avg_perf_OOS[times[i]] = avg_perf_OOS[times[i]] + np.array(estimators_oos[times[i]][j][name]) / len(seeds)
            avg_perf_INS[times[i]] = avg_perf_INS[times[i]] + np.array(estimators_ins[times[i]][j][name]) / len(seeds)

        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2

        ax[a_0, a_1].plot(shrinkage_list, true_values[times[i]])
        ax[a_0, a_1].plot(shrinkage_list, avg_perf_OOS[times[i]])
        ax[a_0, a_1].plot(shrinkage_list, avg_perf_INS[times[i]])
        ax[a_0, a_1].set_title(f'T = {times[i]}')

    ax[0, 0].legend([' INS ', ' OOS ', 'Theoretical '], loc='upper right')
    fig.suptitle(f'c = {c}')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    plt.show()


def expanding_window_experment(t: int,
                               c: float,
                               train_frac: float,
                               beta_and_psi_link: float,
                               shrinkage_list: np.ndarray,
                               seeds: list):
    estimators_oos = []
    estimated_values_saved = []
    [estimated_values_saved.append(run_loo(t=t,
                                           c=c,
                                           train_frac=train_frac,
                                           beta_and_psi_link=beta_and_psi_link,
                                           shrinkage_list=shrinkage_list,
                                           seed=s,
                                           growing_oos=True)) for s in seeds]
    [estimators_oos.append(estimated_values_saved[j][1]) for j in range(len(seeds))]
    # true_values = estimated_values_saved[0][2].theoretical_mean()
    ax_legend = []
    [ax_legend.append(f'Seed {s}') for s in seeds]
    name = 'mean'
    times = estimated_values_saved[0][2].times
    num_seeds = len(estimators_oos)
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):
        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        [ax[a_0, a_1].plot(shrinkage_list, estimators_oos[j][name][i])
         for j in range(num_seeds)]

        ax[a_0, a_1].set_title(f'T_1 = {times[i]}')

    ax[0, 0].legend(ax_legend, loc='upper right')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    fig.suptitle(name.upper() + ' Out of Sample', fontsize=12)
    plt.show()

    return estimators_oos


def ins_vs_oos_experiment(times: list,
                          complexity: list,
                          train_frac: float,
                          beta_and_psi_link: float,
                          shrinkage_list: np.ndarray):
    ax_legend = []
    [ax_legend.extend([f'Complexity = {c} INS', f'Complexity = {c} OOS']) for c in complexity]

    estimators = {}
    for i in range(len(times)):
        estimators[times[i]] = []
        [estimators[times[i]].append(run_loo(t=times[i],
                                             c=c,
                                             train_frac=train_frac,
                                             beta_and_psi_link=beta_and_psi_link,
                                             shrinkage_list=shrinkage_list)[0:2]) for c in complexity]

    for name in estimators[times[0]][0][0].keys():
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
        for i in range(len(times)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][0][name])
             for j in range(len(complexity))]

            [ax[a_0, a_1].plot(shrinkage_list, estimators[times[i]][j][1][name])
             for j in range(len(complexity))]

            ax[a_0, a_1].set_title(f'T = {times[i]}')

        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'Shrinkage Size', ha='center', fontsize=12)
        fig.suptitle(name.upper() + f" \n  beta-psi link= {beta_and_psi_link}", fontsize=12)
        plt.show()
        return estimators


def complete_comparison_experiment(t: int,
                                   c: float,
                                   train_frac: float,
                                   beta_and_psi_link: float,
                                   shrinkage_list: np.ndarray):
    mean_dict = {}
    ins, oos, loo = run_loo(t=t, c=c,
                            train_frac=train_frac,
                            beta_and_psi_link=beta_and_psi_link,
                            shrinkage_list=shrinkage_list,
                            simple_beta=True)

    mean_dict['INS'] = ins['mean']
    mean_dict['OOS'] = oos['mean']

    mean_dict['Theoretical'] = loo.theoretical_mean()

    loo.True_value_eq_176(DataUsed.INS)
    mean_dict['eq 176 INS'] = loo.true_value_sigma_beta_eq_176
    mean_dict['eq 176 limit INS'] = loo.true_value_beta_eq_176

    loo.True_value_eq_176(DataUsed.OOS)
    mean_dict['eq 176 OOS'] = loo.true_value_sigma_beta_eq_176
    mean_dict['eq 176 limit OOS'] = loo.true_value_beta_eq_176

    return


def simulate_across_seeds(t: int,
                          c: float,
                          train_frac: float,
                          beta_and_psi_link: float,
                          shrinkage_list: np.ndarray,
                          seeds: list):
    estimators_oos = []
    estimators_ins = []
    estimated_values_saved = []
    [estimated_values_saved.append(run_loo(t=t, c=c, train_frac=train_frac,
                                           beta_and_psi_link=beta_and_psi_link,
                                           shrinkage_list=shrinkage_list,
                                           seed=s,
                                           simple_beta=True)) for s in seeds]
    [estimators_ins.append(estimated_values_saved[j][0]) for j in range(len(seeds))]
    [estimators_oos.append(estimated_values_saved[j][1]) for j in range(len(seeds))]
    true_values = estimated_values_saved[0][2].theoretical_mean()

    return estimators_ins, estimators_oos, true_values


def across_seeds_experiment(times: list,
                            c: float,
                            train_frac: float,
                            beta_and_psi_link: float,
                            shrinkage_list: np.ndarray,
                            seeds: list):
    estimators_oos = {}
    estimators_ins = {}
    true_values = {}

    for t in times:
        estimators_oos[t], estimators_ins[t], true_values[t] = simulate_across_seeds(t,
                                                                                     c,
                                                                                     train_frac,
                                                                                     beta_and_psi_link,
                                                                                     shrinkage_list,
                                                                                     seeds)
    seeds_legend =[]
    [seeds_legend.append(f'seed {s}') for s in seeds]

    plot_unconditionally(estimators_oos=estimators_oos,
                         estimators_ins=estimators_ins,
                         true_values=true_values,
                         shrinkage_list=shrinkage_list,
                         c=c)

    plot_sub_plots(estimators=estimators_oos,
                   shrinkage_list=shrinkage_list,
                   ax_legend=seeds_legend,
                   c=c,
                   plot_name='Out of Sample')

    plot_sub_plots(estimators=estimators_ins,
                   shrinkage_list=shrinkage_list,
                   ax_legend=seeds_legend,
                   c=c,
                   plot_name='In Sample')



if __name__ == '__main__':
    # testing leave one out:
    times = [20, 250, 1000, 2500]
    # complexity = [0.1, 0.5, 1, 2, 5, 10]
    # times = [10, 50, 100, 250]
    complexity = [0.2, 1, 2.5]
    train_frac = 0.2
    shrinkage_list = np.linspace(0.1, 10, 25)
    beta_and_psi_link = 2
    seeds = list(range(0, 10))
