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
from parameters import *
from ploting import *


def m_ins_vs_oos_plots(t: int,
                       complexity: list,
                       train_frac: float,
                       estimators: list,
                       shrinkage_list: np.ndarray,
                       name: str = 'sharpe',
                       title: str = None):
    ax_legend = []

    [ax_legend.append(f'c = {np.round(c / train_frac, 2)} INS') for c in complexity]
    [ax_legend.append(f'c = {np.round(c / train_frac, 2)} OOS') for c in complexity]

    # for name in list(estimators[times[0]][0].ins_perf_est.keys()):
    fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 8))
    for i in range(4):
        a_0 = i % 2
        a_1 = int((i - a_0) / 2) % 2
        if name == 'sharpe':
            [ax[a_0, a_1].plot(shrinkage_list, estimators[j].ins_m_sharpe[i, :])
             for j in range(len(complexity))]

            [ax[a_0, a_1].plot(shrinkage_list, estimators[j].oos_m_sharpe[i, :])
             for j in range(len(complexity))]

        if name == 'mse':
            [ax[a_0, a_1].plot(shrinkage_list, estimators[j].ins_m_mse[i, :])
             for j in range(len(complexity))]

            [ax[a_0, a_1].plot(shrinkage_list, estimators[j].oos_m_mse[i, :])
             for j in range(len(complexity))]

        ax[a_0, a_1].set_title(f'z{i}={np.round(shrinkage_list[i],2)}')

    ax[0, 0].legend(ax_legend, loc='upper right')
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=12)
    if title is not None:
        fig.suptitle(f'T = T_1 = {int(t / 2)}', fontsize=12)
    plt.show()

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
            simple_beta: bool = False) -> object:
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
    lo_est.ins_performance()
    lo_est.oos_performance()
    lo_est.theoretical_mean_var()

    return lo_est


if __name__ == '__main__':
    # testing leave one out:
    # times = [100, 500, 2000, 4000]
    # complexity = [0.1, 0.5, 1, 2, 5, 10]
    times = [10, 50, 100, 250]
    complexity = [0.2, 1, 2.5]
    train_frac = 0.5
    shrinkage_list = np.linspace(0.1, 10, 100)
    beta_and_psi_link = 2
    seeds = list(range(0, 10))
    # complexity = np.linspace(0., 1, 5)

    estimators = []
    t = 4000

    [estimators.append(run_loo(t=t,
                               c=c,
                               train_frac=train_frac,
                               beta_and_psi_link=beta_and_psi_link,
                               shrinkage_list=shrinkage_list)) for c in complexity]

    m_ins_vs_oos_plots(t=t,
                       complexity=complexity,
                       train_frac=train_frac,
                       estimators=estimators,
                       shrinkage_list=shrinkage_list,
                       name='mse')

    # optimal_vs_oos_plots(times=times,
    #                  complexity=complexity,
    #                  train_frac=train_frac,
    #                  estimators=estimators,
    #                  shrinkage_list=shrinkage_list,
    #                  name='sharpe')
    #
    # optimal_vs_oos_plots(times=times,
    #                  complexity=complexity,
    #                  train_frac=train_frac,
    #                  estimators=estimators,
    #                  shrinkage_list=shrinkage_list,
    #                  name='mse')
    #
    # from colorspacious import cspace_converter
    # import matplotlib as mpl
    #
    # cmaps = {}
    #
    # gradient = np.linspace(0, 1, 256)
    # gradient = np.vstack((gradient, gradient))
    #
    # def plot_color_gradients(category, cmap_list):
    #     # Create figure and adjust figure height to number of colormaps
    #     nrows = len(cmap_list)
    #     figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    #     fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    #     fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
    #                         left=0.2, right=0.99)
    #     axs[0].set_title(f'{category} colormaps', fontsize=14)
    #
    #     for ax, name in zip(axs, cmap_list):
    #         ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])
    #         ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
    #                 transform=ax.transAxes)
    #
    #     # Turn off *all* ticks & spines, not just the ones with colormaps.
    #     for ax in axs:
    #         ax.set_axis_off()
    #
    #     # Save colormap list for later.
    #     cmaps[category] = cmap_list

    # loo = run_loo(t=t,
    #               c=c,
    #               train_frac=train_frac,
    #               shrinkage_list=shrinkage_list)
    #
    # train_sample_length = int(train_frac*t)
    # true_c = np.round(c/train_frac,2)

    # plt.title(f'Mean \n '
    #           f'T = {train_sample_length} \n '
    #           f'c = {true_c}')
    # plt.plot(loo.shrinkage_list, loo.mean_true)
    # plt.plot(loo.shrinkage_list, loo.ins_perf_est['mean'])
    # plt.legend(['Theoretical', 'Empirical'])
    # plt.xlabel('z')
    # plt.show()

    # plt.title(f'Variance \n '
    #           f'T = {train_sample_length} \n '
    #           f'c = {true_c}')
    # plt.plot(loo.shrinkage_list, loo.oos_perf_est['var'])
    # plt.plot(loo.shrinkage_list, loo.ins_perf_est['var'])
    # plt.legend(['OOS', 'INS'])
    # plt.xlabel('z')
    # plt.show()

    # plt.title(f'Mean \n '
    #           f'T = {train_sample_length} \n '
    #           f'c = {true_c}')
    # plt.plot(loo.shrinkage_list, loo.oos_perf_est['mean'])
    # plt.plot(loo.shrinkage_list, loo.ins_perf_est['mean'])
    # plt.legend(['OOS', 'INS'])
    # plt.xlabel('z')
    # plt.show()

    # plt.title(f'Variance \n '
    #           f'T = {train_sample_length} \n '
    #           f'c = {true_c}')
    # plt.plot(loo.shrinkage_list, loo.var_true)
    # plt.plot(loo.shrinkage_list, loo.ins_perf_est['var'])
    # plt.legend(['Theoretical', 'Empirical'])
    # plt.xlabel('z')
    # plt.show()

    # 1 + xi should be the denominator: do test
    # LeaveOut.xi_k_true(
    #
    # )

    # print('Optimal', loo.oos_optimal_mse)
    # print('regular', np.array(loo.oos_perf_est['mse']).mean())
    # print('Optimal', loo.oos_optimal_sharpe)
    # print('regular', np.array(loo.oos_perf_est['sharpe']).mean())
    #
    # ones = np.ones([len(shrinkage_list), 1])
    #
    # plt.title(f'MSE \n c = {c}')
    # plt.plot(shrinkage_list, loo.oos_perf_est['mse'])
    # plt.plot(shrinkage_list, ones * loo.oos_optimal_mse)
    # plt.legend(['Overall', 'Optimal'])
    # plt.xlabel('z')
    # plt.show()
    #
    # plt.title(f'sharpe \n c = {c}')
    # plt.plot(shrinkage_list, loo.oos_perf_est['sharpe'])
    # plt.plot(shrinkage_list, ones * loo.oos_optimal_sharpe)
    # plt.legend(['Overall', 'Optimal'])
    # plt.xlabel('z')
    # plt.show()
