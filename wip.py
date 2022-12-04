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
from ploting import *


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
    lo_est.ins_performance()
    lo_est.oos_performance()

    return lo_est


if __name__ == '__main__':
    # testing leave one out:
    times = [20, 250, 1000, 2500]
    # complexity = [0.1, 0.5, 1, 2, 5, 10]
    # times = [10, 50, 100, 250]
    # complexity = [0.2, 1, 2.5]
    train_frac = 0.7
    shrinkage_list = np.linspace(0.1, 10, 100)
    beta_and_psi_link = 2
    seeds = list(range(0, 10))
    complexity = np.linspace(0.2, 1, 5)
    t = 2000

    for c in complexity:

        loo = run_loo(t=t,
                      c=c,
                      train_frac=train_frac,
                      shrinkage_list=shrinkage_list)

        print('Optimal', loo.oos_optimal_mse)
        print('regular', np.array(loo.oos_perf_est['mse']).mean())
        print('Optimal', loo.oos_optimal_sharpe)
        print('regular', np.array(loo.oos_perf_est['sharpe']).mean())

        ones = np.ones([len(shrinkage_list), 1])

        plt.title(f'MSE \n c = {c}')
        plt.plot(shrinkage_list, loo.oos_perf_est['mse'])
        plt.plot(shrinkage_list, ones * loo.oos_optimal_mse)
        plt.legend(['Overall', 'Optimal'])
        plt.show()

        plt.title(f'sharpe \n c = {c}')
        plt.plot(shrinkage_list, loo.oos_perf_est['sharpe'])
        plt.plot(shrinkage_list, ones * loo.oos_optimal_sharpe)
        plt.legend(['Overall', 'Optimal'])
        plt.show()

    #
    # ret_mat_for_z = np.array(loo.ret_vec_ins).reshape(-1,len(loo.shrinkage_list))
    # v = np.sum(ret_mat_for_z,0)
    # pi_mat_for_z = np.array(loo.pi_ins).reshape(-1,len(loo.shrinkage_list))
    # m_mse = pi_mat_for_z.T @ pi_mat_for_z
    # m_sharpe = ret_mat_for_z.T @ ret_mat_for_z
    # w_mse = np.linalg.pinv(m_mse) @ v
    # w_sharpe = np.linalg.pinv(m_sharpe) @ v
    # beta_hat_mat = np.array(loo.beta_hat).reshape(loo.p,-1)
    # beta_hat_optimal_sharpe = beta_hat_mat @ w_sharpe
    # beta_hat_optimal_mse = beta_hat_mat @ w_mse
