import numpy
import numpy as np
import pandas as pd
from leave_out import *
from main import *
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.marcenko_pastur import MarcenkoPastur


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

    W = leave_out.smart_w_matrix(features=features,
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
    m_list = leave_out.empirical_stieltjes(eigenvalues, P, shrinkage_list)
    mean_estimator = [(1 - c + c * shrinkage_list[i] * m_list[i]) * estimator_list[i] for i in
                      range(len(shrinkage_list))]

    return mean_estimator


def run_loo(t: int,
            c: float,
            train_frac: float,
            seed: int = None,
            beta_and_psi_link: float = None,
            shrinkage_list: list = None,
            true_values: bool = False) -> object:
    seed = 0 if seed is None else seed

    lo_est = leave_out(t, c)
    lo_est.seed = seed

    beta_and_psi_link = lo_est.beta_and_psi_link if beta_and_psi_link is None else beta_and_psi_link
    shrinkage_list = lo_est.shrinkage_list if shrinkage_list is None else shrinkage_list

    lo_est.beta_and_psi_link = beta_and_psi_link
    lo_est.shrinkage_list = shrinkage_list
    lo_est.simulate_date()
    lo_est.train_test_split(train_frac)
    lo_est.train_model()
    estimator_in_sample = lo_est.ins_performance()
    estimator_out_of_sample = lo_est.oos_performance()
    if true_values:
        lo_est.calculate_true_value()

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
        fig.text(0.5, 0.04, 'Shrinkage Size', ha='center', fontsize=12)
        fig.suptitle(name.upper() + ' ' + plot_name + f" \n  Complexity = {c}", fontsize=12)
        plt.show()


if __name__ == '__main__':
    # testing leave one out:
    # times = [100, 500, 1000, 2500]
    # complexity = [0.1, 0.5, 1, 2, 5, 10]
    times = [10, 50, 100, 250]
    complexity = [0.2, 5]
    train_frac = 0.5
    shrinkage_list = np.linspace(0.1, 10, 100)
    beta_and_psi_link = 2
    seeds = list(range(0, 10))

    #
    # # ins vs oos
    # ax_legend = []
    # [ax_legend.extend([f'Complexity = {c} INS', f'Complexity = {c} OOS']) for c in complexity]
    #
    # estimators_oos = {}
    # for i in range(len(times)):
    #     estimators_oos[times[i]] = []
    #     [estimators_oos[times[i]].append(run_loo(t=times[i],
    #                                          c=c,
    #                                          train_frac=train_frac,
    #                                          beta_and_psi_link=beta_and_psi_link,
    #                                          shrinkage_list=shrinkage_list)[0:2]) for c in complexity]
    #
    # for name in estimators_oos[times[0]][0][0].keys():
    #     fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    #     for i in range(len(times)):
    #         a_0 = i % 2
    #         a_1 = int((i - a_0) / 2) % 2
    #         [ax[a_0, a_1].plot(shrinkage_list, estimators_oos[times[i]][j][0][name])
    #         for j in range(len(complexity))]
    #
    #         [ax[a_0, a_1].plot(shrinkage_list, estimators_oos[times[i]][j][1][name])
    #         for j in range(len(complexity))]
    #
    #         ax[a_0, a_1].set_title(f'T = {times[i]}')
    #
    #
    #     ax[0, 0].legend(ax_legend, loc='upper right')
    #     fig.text(0.5, 0.04, 'Shrinkage Size', ha='center', fontsize=12)
    #     fig.suptitle(name.upper() + f" \n  beta-psi link= {beta_and_psi_link}", fontsize=12)
    #     plt.show()

    # OOS across seeds

    ax_legend = []
    [ax_legend.append(f'Seed {s}') for s in seeds]
    c = 5
    estimators_oos = {}
    estimators_ins = {}
    true_values = {}
    for i in range(len(times)):
        estimators_oos[times[i]] = []
        estimators_ins[times[i]] = []
        true_values[times[i]] = []
        estimated_values_saved = []
        [estimated_values_saved.append(run_loo(t=times[i], c=c, train_frac=train_frac,
                                               beta_and_psi_link=beta_and_psi_link,
                                               shrinkage_list=shrinkage_list,
                                               seed=s,
                                               true_values=True)) for s in seeds]
        [estimators_ins[times[i]].append(estimated_values_saved[j][0]) for j in range(len(seeds))]
        [estimators_oos[times[i]].append(estimated_values_saved[j][1]) for j in range(len(seeds))]
        [true_values[times[i]].append(estimated_values_saved[j][2].true_value_mean) for j in range(len(seeds))]

    plot_sub_plots(estimators=estimators_oos,
                   shrinkage_list=shrinkage_list,
                   ax_legend=ax_legend,
                   c=c,
                   plot_name='Out of Sample')

    plot_sub_plots(estimators=estimators_ins,
                   shrinkage_list=shrinkage_list,
                   ax_legend=ax_legend,
                   c=c,
                   plot_name='In Sample')

    # compare with true value

    name = 'mean'
    avg_perf_OOS = {}
    avg_perf_INS = {}
    avg_true_value = {}
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(len(times)):
        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        avg_perf_OOS[times[i]] = np.zeros(len(estimators_oos[times[i]][0][name]))
        avg_perf_INS[times[i]] = np.zeros(len(estimators_ins[times[i]][0][name]))
        avg_true_value[times[i]] = np.zeros(len(true_values[times[i]][0]))
        for j in range(len(seeds)):
            avg_perf_OOS[times[i]] = avg_perf_OOS[times[i]] + np.array(estimators_oos[times[i]][j][name]) / len(seeds)
            avg_perf_INS[times[i]] = avg_perf_INS[times[i]] + np.array(estimators_ins[times[i]][j][name]) / len(seeds)
            avg_true_value[times[i]] = avg_true_value[times[i]] + np.array(true_values[times[i]][j]) / len(seeds)

        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        ax[a_0, a_1].plot(shrinkage_list, avg_perf_OOS[times[i]])
        ax[a_0, a_1].plot(shrinkage_list, avg_perf_INS[times[i]])
        ax[a_0, a_1].plot(shrinkage_list, avg_true_value[times[i]])
        ax[a_0, a_1].set_title(f'T = {times[i]}')

    ax[0, 0].legend(['Average INS Value ', 'Average OOS Value', 'Theoretical Value'], loc='upper right')
    fig.text(0.5, 0.04, 'Shrinkage Size', ha='center', fontsize=12)
    fig.suptitle(name.upper() + f" \n  Complexity = {c}", fontsize=12)
    plt.show()
