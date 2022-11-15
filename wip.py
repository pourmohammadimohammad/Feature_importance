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

def smart_eigenvalue_decomposition(features: np.ndarray,
                                   T: int = None):
    """

    :param features: features used to create covariance matrix T x P
    :param T: Weight used to normalize matrix
    :return: Left eigenvectors PxT and eigenvalues without zeros
    """
    [T_true, P] = features.shape
    T = T_true if T is None else T

    if P > T:
        print('complex regime')
        covariance = features @ features.T / T

    else:
        print('regular regime')
        covariance = features.T @ features / T

    eigval, eigvec = np.linalg.eigh(covariance)
    eigvec = eigvec[:, eigval > 10 ** (-10)]
    eigval = eigval[eigval > 10 ** (-10)]

    if P > T:
        eigvec = np.matmul(features.T, eigvec * ((eigval * T) ** (-1 / 2)).reshape(1, -1))

    return eigval, eigvec


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
    [T, P] = features.shape
    projected_features = eigenvectors.T @ features.T / np.sqrt(T)
    stuff_divided = [(1 / (eigenvalues.reshape(-1, 1) + z)) * projected_features for z in shrinkage_list]

    W = [projected_features.T @ x_ for x_ in stuff_divided]

    cov_left_over = features @ features.T / T - projected_features.T @ projected_features

    W_list = [W[i] + (1 / shrinkage_list[i]) * cov_left_over for i in range(len(shrinkage_list))]
    return W_list


def leave_two_out_estimator_vectorized(labels: np.ndarray,
                                       features: np.ndarray,
                                       eigenvalues: np.ndarray,
                                       eigenvectors: np.ndarray,
                                       shrinkage_list: np.ndarray) -> float:
    """
    # Implement leave two out estimators
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

    W = smart_w_matrix(features=features,
                       eigenvalues=eigenvalues,
                       eigenvectors=eigenvectors,
                       shrinkage_list=shrinkage_list)

    T = np.shape(features)[0]

    num = (T - 1)  # divded by T to account for W normalization

    labels_squared = (labels.reshape(-1, 1) * np.squeeze(labels)).reshape(1, -1)

    W_diag = [(1 - np.diag(w)).reshape(-1, 1) * (1 - np.diag(w)) for w in W]
    [np.fill_diagonal(w, 0) for w in W]  # get rid of diagonal elements not in the calculations
    estimator_list = [
        np.sum(labels_squared * W[i].reshape(1, -1) / (W_diag[i].reshape(1, -1) - W[i].reshape(1, -1) ** 2)) / num \
        for i in range(len(shrinkage_list))]

    return estimator_list


def leave_one_out_estimator(labels: np.ndarray,
                            features: np.ndarray,
                            eigenvalues: np.ndarray,
                            eigenvectors: np.ndarray,
                            shrinkage_list: np.ndarray) -> float:
    """
    # Implement leave one out estimator

    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param eigenvectors:
    :param eigenvalues:
    :param shrinkage_list:
    :return: Unbiased estimator
    """

    W = smart_w_matrix(features=features,
                       eigenvalues=eigenvalues,
                       eigenvectors=eigenvectors,
                       shrinkage_list=shrinkage_list)

    W_diag_trans = [(1 / (1 - np.diag(w)) - 1).reshape(-1, 1) * (labels.reshape(-1, 1) ** 2) for w in W]
    s_beta = [
        (1 / (1 - np.diag(w))).reshape(-1, 1) * labels.reshape(-1, 1) * (labels.reshape(1, -1) * w).sum(0).reshape(-1,
                                                                                                                   1)
        for w in W]
    estimator_list = [np.mean(W_diag_trans[i] - s_beta[i]) for i in range(len(shrinkage_list))]
    return estimator_list


if __name__ == '__main__':
    # testing leave one out and leave two out:

    seed = 0
    sample_size = 1000
    number_features_ = 10000
    beta_and_psi_link_ = 2
    noise_size_ = 0
    activation_ = 'linear'
    number_neurons_ = 1
    shrinkage_list = np.linspace(0.1, 10, 100)

    labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
                                                                 sample_size=sample_size,
                                                                 number_features_=number_features_,
                                                                 beta_and_psi_link_=beta_and_psi_link_,
                                                                 noise_size_=noise_size_,
                                                                 activation_=activation_,
                                                                 number_neurons_=number_neurons_)

    eigenvalues, eigenvectors = smart_eigenvalue_decomposition(features)

    estimator_list = leave_two_out_estimator_vectorized(labels, features, eigenvalues, eigenvectors, shrinkage_list)
    psi_beta = psi_eigenvalues.reshape(1, -1) * beta_dict[0].reshape(1, -1)
    eigenvectors_projection = psi_beta @ eigenvectors
    true_values = [np.sum((1 / (eigenvalues.reshape(1, -1) + z)) * eigenvectors_projection ** 2) for z in
                   shrinkage_list]
    left_over = np.sum(psi_beta ** 2) - np.sum(eigenvectors_projection ** 2)
    true_values = [true_values[i] + (1 / shrinkage_list[i]) * left_over for i in range(len(shrinkage_list))]

    plt.plot(shrinkage_list, true_values)
    plt.plot(shrinkage_list, estimator_list)
    plt.legend(['True Value', 'Estimator'])
    plt.title(f'Leave two out \n'
              f'Error for P = {number_features_}, T = {sample_size} \n beta_and_psi_link_ = {beta_and_psi_link_}')
    plt.xlabel('Value of z, shrinkage')
    plt.show()

    # very_true_values = [ (psi_beta.reshape(1,-1) @ np.linalg.pinv(z*np.eye(number_features_) + features.T@features/sample_size) \
    #                      @psi_beta.reshape(-1,1))[0] for z in shrinkage_list]
    #
    # empirical_stieltjes = (1 / (eigenvalues.reshape(-1, 1) + shrinkage_list.reshape(1, -1))).sum(0) / P \
    #                       + (P - len(eigenvalues)) / shrinkage_list / P
