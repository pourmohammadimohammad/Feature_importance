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

def map_w_to_w_tilde(w_matrix):
    """
    We need to map W_{t_1,t_2} to W_{t_1,t_2}/((1-W_{t1,t1})(1-W_{t2,t2})-W_{t1,t2}^2)
    :param w_matrix:
    :return:
    """
    diag = (1 - np.diag(w_matrix)).reshape(-1, 1) * (1 - np.diag(w_matrix))
    denominator = diag - (w_matrix ** 2)
    w_tilde = w_matrix / denominator
    return np.fill_diagonal(w_tilde, 0)


def leave_two_out_estimator_vectorized_resolvent(labels: np.ndarray,
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

    num = T * (T - 1)  # divided by T to account for W normalization
    labels = labels.reshape(-1, 1)

    estimator_list = [labels.T @ map_w_to_w_tilde(w_matrix) @ labels / num for w_matrix in W]

    return estimator_list


def leave_two_out_estimator_vectorized_general(labels: np.ndarray,
                                               features: np.ndarray,
                                               A: np.ndarray) -> float:
    """
    # Implement leave two out estimators
    # For any matrix A independent of t_1 and t_2
    # E[R_{t_2+1} S_{t_1}'A S_{t_2} R_{t_1+1}]\ =
    \  \beta'\Psi A_ right \Psi \beta\

    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param A: Weighting matrix
    :return: Unbiased estimator
    """

    T = np.shape(features)[0]

    num = (T - 1) * T  # divded by T to account for W normalization

    labels_squared = (labels.reshape(-1, 1) * np.squeeze(labels)).reshape(1, -1)
    matrix_multiplied = features.T @ A @ features
    np.fill_diagonal(matrix_multiplied, 0)

    estimator = np.sum(labels_squared * matrix_multiplied.reshape(1, -1)) / num

    return estimator


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

    labels_squared = (labels.reshape(-1, 1) ** 2).reshape(-1,1)

    normalizer = [
        (1 / (1 - np.diag(w))).reshape(-1, 1) for w in W]

    labels_squared_normalized = [
        labels_squared*(1-n) for n in normalizer]

    s_beta = [normalizer[i]*labels.reshape(-1, 1) * (labels.reshape(1, -1) * W[i]).sum(1).reshape(-1,1) for i in range(len(shrinkage_list))]

    estimator_list = [np.mean(s_beta[i] + labels_squared_normalized[i]) for i in range(len(shrinkage_list))]

    return estimator_list


def leave_one_out_true_value(beta_dict: np.ndarray,
                             psi_eigenvalues: np.ndarray,
                             eigenvalues: np.ndarray,
                             eigenvectors: np.ndarray,
                             shrinkage_list: np.ndarray):
    """
    Efficient way to estimate \beta \Psi (\hat \Psi + zI)^{-1} \Psi \beta
    :param beta_dict: beta paramaters (ground truth)
    :param psi_eigenvalues: True eigenvalues of covariance matrix
    :param eigenvalues: eigenvalues of covariance matrix
    :param eigenvectors: eigenvectors of covariance matrix
    :param shrinkage_list:
    :return:
    """

    psi_beta = psi_eigenvalues.reshape(1, -1) * beta_dict[0].reshape(1, -1)

    eigenvectors_projection_left = (eigenvectors.T @ psi_beta.reshape(-1,1)).reshape(1,-1)
    eigenvectors_projection_right = (beta_dict[0].reshape(1, -1) @ eigenvectors).reshape(1,-1)

    true_values = [
        np.sum((1 / (eigenvalues.reshape(1, -1) + z)) * eigenvectors_projection_left * eigenvectors_projection_right)
        for z in
        shrinkage_list]
    left_over = np.sum(eigenvectors_projection_left * eigenvectors_projection_right)

    true_values = [left_over - shrinkage_list[i] * true_values[i] for i in range(len(shrinkage_list))]

    return true_values


def empirical_stieltjes(eigenvalues, P, shrinkage_list):
    """
    :param eigenvalues: Eigenvalues of covariance matrix
    :param P: Number of features
    :param shrinkage_list: List of shrinkage z
    :return: empirical stieltjes transform of normalized covariance matrix
    """

    estimator = (1 / (eigenvalues.reshape(-1, 1) + shrinkage_list.reshape(1, -1))).sum(0) / P \
                + (P - len(eigenvalues)) / shrinkage_list / P

    return estimator


def efficient_beta_psi_resolvent_true_value(beta_dict: np.ndarray,
                                 psi_eigenvalues: np.ndarray,
                                 eigenvalues: np.ndarray,
                                 eigenvectors: np.ndarray,
                                 shrinkage_list: np.ndarray):
    """
    Efficient way to estimate \beta \Psi (\hat \Psi + zI)^{-1} \Psi \beta
    :param beta_dict: beta paramaters (ground truth)
    :param psi_eigenvalues: True eigenvalues of covariance matrix
    :param eigenvalues: eigenvalues of covariance matrix
    :param eigenvectors: eigenvectors of covariance matrix
    :param shrinkage_list:
    :return:
    """

    psi_beta = psi_eigenvalues.reshape(1, -1) * beta_dict[0].reshape(1, -1)
    eigenvectors_projection = psi_beta @ eigenvectors
    true_values = [np.sum((1 / (eigenvalues.reshape(1, -1) + z)) * eigenvectors_projection ** 2) for z in
                   shrinkage_list]
    left_over = np.sum(psi_beta ** 2) - np.sum(eigenvectors_projection ** 2)
    true_values = [true_values[i] + (1 / shrinkage_list[i]) * left_over for i in range(len(shrinkage_list))]

    return true_values


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

    estimator_list = leave_one_out_estimator(labels, features, eigenvalues, eigenvectors,
                                                                  shrinkage_list)
    true_values = leave_one_out_true_value(beta_dict, psi_eigenvalues, eigenvalues, eigenvectors, shrinkage_list)

    beta_psi_beta = psi_eigenvalues.reshape(1, -1) @ (beta_dict[0].reshape(-1, 1)**2)


    plt.plot(shrinkage_list, true_values)
    plt.plot(shrinkage_list, estimator_list)
    plt.legend(['true_value', 'Estimator'])
    plt.title(f'Leave two out \n'
              f'Error for P = {number_features_}, T = {sample_size} \n beta_and_psi_link_ = {beta_and_psi_link_}')
    plt.xlabel('Value of z, shrinkage')
    plt.show()

    A = np.eye(number_features_)
