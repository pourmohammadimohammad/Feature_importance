import numpy
import numpy as np
import pandas as pd

from main import simulate_data
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.marcenko_pastur import MarcenkoPastur


# mohammad_is_wrong = RandomFeatures.naive_linear_single_underlying()

def smart_w_matrix(features: np.ndarray,
                   eigenvalues: np.ndarray,
                   eigenvectors: np.ndarray,
                   shrinkage_list: np.ndarray):
    """
    (z+Psi)^{-1} = U (z+lambda)^{-1}U' + z^{-1} (I - UU')
    we compute S'(z+Psi)^{-1} S= S' (
    :param features:
    :param eigenvalues:
    :param eigenvectors:
    :param shrinkage_list:
    :return:
    """
    projected_features = eigenvectors.T @ features.T
    stuff_divided = [(1 / (eigenvalues.reshape(-1, 1) + z)) * projected_features for z in shrinkage_list]

    stuff = [projected_features.T @ x_ for x_ in stuff_divided]

    cov = features @ features.T - projected_features.T @ projected_features

    stuff = [stuff[i] + (1 / shrinkage_list[i]) * cov for i in range(shrinkage_list)]
    return stuff



def leave_two_out_estimator_resolvent(labels: np.ndarray,
                                      features: np.ndarray,
                                      z: float,
                                      resolvent: bool = True,
                                      smart_inv: bool = False) -> float:
    """
    # Implement leave two out estimators
    # For any matrix A independent of t_1 and t_2
    # E[R_{t_2+1} S_{t_1}'A S_{t_2} R_{t_1+1}]\ =
    \  \beta'\Psi A_left (\hat \Psi + zI)^{-1} A_right \Psi \beta\

    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param z: Ridge shrinkage
    :param resolvent: Estimate quadratic form with resolvent
    :param smart_inv: Use smart covariance estimation using sprectral decompositoin
    :return: Unbiased estimator
    """
    [T, P] = np.shape(features)

    if resolvent == T:
        if smart_inv == True:
            inv = smart_cov_inv(features, z, T)
        if smart_inv == False:
            inv = np.linalg.pinv(features.T @ features / T + z * np.eye(P))

    estimator = 0
    num = (T - 1) * T / 2

    for i in range(T):
        for j in range(i + 1, T):
            w_ij = features[i, :].reshape(1, -1) @ inv @ features[j, :].reshape(-1, 1) / T
            w_ii = features[i, :].reshape(1, -1) @ inv @ features[i, :].reshape(-1, 1) / T
            w_jj = features[j, :].reshape(1, -1) @ inv @ features[j, :].reshape(-1, 1) / T
            w_estimator = w_ij / ((1 - w_ii) * (1 - w_jj) - w_ij ** 2)
            estimator += T * w_estimator * labels[i] * labels[j]

    estimator = estimator / num
    return estimator


def leave_two_out_estimator_fast(labels: np.ndarray,
                                 features: np.ndarray,
                                 A: np.ndarray) -> float:
    """
    # Implement leave two out estimators
    # For any matrix A independent of t_1 and t_2
    # E[R_{t_2+1} S_{t_1}'A S_{t_2} R_{t_1+1}]\ =\  \beta'\Psi A \Psi \beta\
    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param A: Weighting matrix to help with estimation
    :return: Unbiased estimator
    """
    [sample_size, number_features] = np.shape(features)
    estimator = 0
    num = (sample_size - 1) * sample_size / 2

    if np.prod(A == np.eye(number_features)):
        for i in tqdm(range(sample_size - 1)):
            estimator += labels[i + 1:, :].T @ (features[i + 1:, :] @ features[i, :].reshape(-1, 1)) * labels[i]
    else:
        for i in tqdm(range(sample_size - 1)):
            estimator += labels[i + 1:, :].T @ (features[i + 1:, :] @ A @ features[i, :].reshape(-1, 1)) * labels[i]

    estimator = estimator / num
    return estimator


def leave_one_out(labels: np.ndarray,
                  features: np.ndarray,
                  z: float,
                  smart_inv: bool = False) -> float:
    [T, P] = np.shape(features)

    estimator = 0
    num = (T - 1) * T / 2

    for i in range(T):
        features_t = np.delete(features, i, 0)
        if smart_inv:
            inv = smart_inv(features_t, z, T)
        else:
            inv = np.linalg.pinv(features_t.T @ features_t / T + z * np.eye(P))

        estimator += labels[i] * (
                features[i].reshape(1, -1) @ inv @ features[i + 1:, :].T @ labels[i + 1:].reshape(-1, 1))

    estimator = estimator / num
    return estimator


def smart_eigenvalue_decomposition(features: np.ndarray,
                                   T: int):
    [T_true, P] = features.shape
    covariance = features.T @ features / T

    if T > P:
        print('regular regime')
        eigval, eigvec = np.linalg.eigh(covariance)
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]

    else:
        print('complex regime')
        cov_t = features @ features.T / T
        eigval, eigvec = np.linalg.eigh(cov_t)
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]

        eigvec = np.matmul(features.T, eigvec * ((eigval * T) ** (-1 / 2)).reshape(1, -1))
    return eigval, eigvec


def smart_cov_inv(features: np.ndarray, z: float, T: int) -> np.ndarray:
    """

    :param features: signals
    :param z: ridge shrinkage
    :param T: number of sample
    :return: computationally cheap covariance matrix
    """

    [T_true, P] = features.shape
    covariance = features.T @ features / T

    if T > P:
        print('regular regime')
        inv = eigvec @ np.diag((z + eigval) ** (-1)) @ eigvec.T

    else:
        inv = eigvec @ np.diag((z + eigval) ** (-1)) @ eigvec.T + (np.eye(P) - eigvec @ eigvec.T) * (1 / z)

    return inv


if __name__ == '__main__':
    shrinkage_list = np.exp(np.arange(-10, 10, 5)).tolist()
    c_ = P / T
    mp = MarcenkoPastur.marcenko_pastur(c_, shrinkage_list)
    eigenvalues, eigenvectors = smart_eigenvalue_decomposition(features, T)
    eigenvalues = eigenvalues  # add np.zeros(P - len(eigenvalues))

    empirical_stieltjes = (1 / (eigenvalues.reshape(-1, 1) + shrinkage_list.reshape(1, -1))).sum(0) / P \
                          + (P - len(eigenvalues)) / shrinkage_list / P
    # TODO MOHAMMAD, PLEASE TEST THAT EMPIRICAL STIELTJES COINCIDES WITH MARCENKO-PASTUR


    # run_experiment(seed=0,
    #                sample_size=100,
    #                number_features_=10000,
    #                beta_and_psi_link_=2,
    #                noise_size_=0,
    #                activation_='linear',
    #                number_neurons_=1,
    #                shrinkage_list=np.exp(np.arange(-10, 10, 5)).tolist())

    T = np.linspace(100, 1000, 10)
    P = np.linspace(100, 1000, 10)
    error_beta2_psi2 = []
    error_beta2_psi = []
    for t in T:
        seed = 0
        sample_size = int(t)
        number_features_ = 100
        beta_and_psi_link_ = 2
        noise_size_ = 0
        activation_ = 'linear'
        number_neurons_ = 1
        shrinkage_list = np.exp(np.arange(-10, 10, 5)).tolist()

        labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
                                                                     sample_size=sample_size,
                                                                     number_features_=number_features_,
                                                                     beta_and_psi_link_=beta_and_psi_link_,
                                                                     noise_size_=noise_size_,
                                                                     activation_=activation_,
                                                                     number_neurons_=number_neurons_)
        A = np.eye(number_features_)

        estimator = leave_two_out_estimator_fast(labels, features, A)
        true_value = psi_eigenvalues ** 2 @ beta_dict[0] ** 2
        percentage_error = np.abs(estimator - true_value) / true_value

        estimator_21 = np.mean(labels ** 2)
        true_value_21 = psi_eigenvalues @ beta_dict[0] ** 2
        percentage_error_21 = np.abs(estimator_21 - true_value_21) / true_value_21
        error_beta2_psi.append(percentage_error_21[0])

        error_beta2_psi2.append(percentage_error[0])

    plt.plot(T, error_beta2_psi2)
    plt.title(f'Error for fixed P = {number_features_} increasing T')
    plt.ylabel('Error in percentage')
    plt.xlabel('Value of T')
    plt.show()

    plt.plot(T, error_beta2_psi)
    plt.title(f'Error for fixed P = {number_features_} increasing T')
    plt.ylabel('Error in percentage')
    plt.xlabel('Value of T')
    plt.show()

    error_beta2_psi2 = []
    error_beta2_psi = []

    for p in P:
        seed = 0
        sample_size = 100
        number_features_ = int(p)
        beta_and_psi_link_ = 2
        noise_size_ = 0
        activation_ = 'linear'
        number_neurons_ = 1
        shrinkage_list = np.exp(np.arange(-10, 10, 5)).tolist()

        labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
                                                                     sample_size=sample_size,
                                                                     number_features_=number_features_,
                                                                     beta_and_psi_link_=beta_and_psi_link_,
                                                                     noise_size_=noise_size_,
                                                                     activation_=activation_,
                                                                     number_neurons_=number_neurons_)
        A = np.eye(number_features_)

        estimator = leave_two_out_estimator_fast(labels, features, A)
        true_value = psi_eigenvalues ** 2 @ beta_dict[0] ** 2
        percentage_error = np.abs(estimator - true_value) / true_value

        estimator_21 = np.mean(labels ** 2)
        true_value_21 = psi_eigenvalues @ beta_dict[0] ** 2
        percentage_error_21 = np.abs(estimator_21 - true_value_21) / true_value_21
        error_beta2_psi.append(percentage_error_21[0])

        error_beta2_psi2.append(percentage_error[0])

    plt.plot(P, error_beta2_psi2)
    plt.title(f'Error for fixed T = {sample_size} increasing P')
    plt.ylabel('Error in percentage')
    plt.xlabel('Value of P')
    plt.show()

    plt.plot(P, error_beta2_psi)
    plt.title(f'Error for fixed T = {sample_size} increasing P')
    plt.ylabel('Error in percentage')
    plt.xlabel('Value of P')
    plt.show()
