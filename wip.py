import numpy
import numpy as np
import pandas as pd

from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm


def simulate_data(seed: int,
                  sample_size: int,
                  number_features_: int,
                  beta_and_psi_link_: float,
                  noise_size_: float,
                  activation_: str = 'linear',
                  number_neurons_: int = 1
                  ):
    """
    this function simulates potentially highly non-linear data on which we will be testing
    our RMT stuff

    :param number_neurons_: we generate data y = \sum_{i=1}^K activation(features * beta + noise)
    each summand is a neuron. K = number_neurons #Mohammad: How are we summing this stuff? so we use different random
    activation functions of the same class?
    :param activation_: type of non-linearity  #Mohammad: Do we need to specify parameters for the functions we use?
    :param sample_size: sample size
    :param number_features_: number_features #Mohammad: Low dimensional features? That can become high dimensional
    through our activation functions
    :param beta_and_psi_link_: how eigenvalues of Sigma_beta and Psi are related #Mohammad: This the crucial
    parameter for our feature importance application
    :param noise_size_: size of noise
    :return: labels and features. Labels are noisy and potentially non-linear functions of features
    """
    np.random.seed(seed)
    psi_eigenvalues = np.abs(np.random.uniform(0.01, 1, [1, number_features_]))
    features = np.random.randn(sample_size, number_features_) * (psi_eigenvalues ** 0.5)

    # should also divide by P to ensure bounded trace norm
    beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_ /number_features_  # we should also experiment with non-monotonic links
    labels_ = np.zeros([sample_size, 1])
    beta_dict = dict()
    for neuron in range(number_neurons_):
        betas = np.random.randn(number_features_, 1) * (beta_eigenvalues ** 0.5).reshape(-1, 1)
        noise = np.random.randn(sample_size, 1) * noise_size_

        labels_ \
            += RandomFeaturesGenerator.apply_activation_to_multiplied_signals(
            multiplied_signals=features @ betas + noise,
            activation=activation_)
        beta_dict[neuron] = betas
    return labels_, features, beta_dict, psi_eigenvalues


def run_experiment(seed: int,
                   sample_size: int,
                   number_features_: int,
                   beta_and_psi_link_: float,
                   noise_size_: float,
                   shrinkage_list: list,
                   activation_: str = 'linear',
                   number_neurons_: int = 1,
                   use_random_features: bool = False,

                   ):
    """

    :param shrinkage_list:
    :param seed:
    :param sample_size:
    :param number_features_:
    :param beta_and_psi_link_:
    :param noise_size_:
    :param activation_:
    :param number_neurons_:
    :param use_random_features:
    :return: Plot for comparison
    """

    labels, features, beta_dict = simulate_data(seed=seed,
                                                sample_size=sample_size,
                                                number_features_=number_features_,
                                                beta_and_psi_link_=beta_and_psi_link_,
                                                noise_size_=noise_size_,
                                                activation_=activation_,
                                                number_neurons_=number_neurons_)
    gamma = 1.
    number_random_features = 10000

    specification = {'distribution': 'normal',
                     'distribution_parameters': [0, gamma],
                     'activation': activation_,
                     'number_features': number_random_features,
                     'bias_distribution': None,
                     'bias_distribution_parameters': [0, gamma]}
    if use_random_features:
        random_features = RandomFeaturesGenerator.generate_random_neuron_features(
            features,
            seed + 10,
            **specification
        )
    else:
        random_features = features

    in_sample_period = int(sample_size / 2)

    regression_results = RandomFeatures.ridge_regression_single_underlying(
        signals=features[:in_sample_period, :],
        labels=labels[:in_sample_period],
        future_signals=features[in_sample_period:, :],
        shrinkage_list=shrinkage_list,
        use_msrr=False,
        return_in_sample_pred=True,
        compute_smart_weights=False,
        test_fixed_kappas=False,
        fixed_kappas=[],
        test_changing_kappas=False,
        constant_signal=None,
        print_time=False,
        return_beta=True,
        keep_only_big_beta=False,
        core_z_values=None,
        clip_bstar=10000)  # understand what this clip does

    plt.title(['Beta for different shrinkage and c=' + str(number_features_ / sample_size)])
    legend_list = []
    for i in range(len(shrinkage_list)):
        legend_list.append(['z = ' + str(round(shrinkage_list[i], 2))])
        plt.scatter(regression_results['betas'][i], beta_dict[0])

    plt.legend(legend_list)
    plt.xlabel('True Beta')
    plt.ylabel('Estimated Beta')
    plt.show()


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
            inv = smart_inv(features, z, T)
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

        estimator += labels[i] * (features[i].reshape(1, -1) @ inv @ features[i + 1:, :].T @ labels[i + 1:].reshape(-1,1))

    estimator = estimator / num
    return estimator


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
        eigval, eigvec = np.linalg.eigh(covariance)
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]
        inv = eigvec @ np.diag((z + eigval) ** (-1)) @ eigvec.T

    else:
        print('complex regime')
        cov_t = features @ features.T / T
        eigval, eigvec = np.linalg.eigh(cov_t)
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]
        eigvec = np.matmul(features.T, eigvec * ((eigval * T) ** (-1 / 2)).reshape(1, -1))
        inv = eigvec @ np.diag((z + eigval) ** (-1)) @ eigvec.T

    return inv


if __name__ == '__main__':

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



