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
from wip import *


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


def very_true_values_xi(features: np.ndarray,
                        beta_dict: np.ndarray,
                        psi_eigenvalues: np.ndarray,
                        shrinkage_list: np.ndarray):
    [T, P] = features.shape
    inverse = [np.linalg.pinv(features.T @ features / T + z * np.eye(P)) for z in shrinkage_list]
    inverse_squared = [i @ i for i in inverse]
    xi_beta_true = [(beta_dict[0].T @ (inverse[i] - shrinkage_list[i] * inverse_squared[i]) @ beta_dict[0])[0]
                    for i in range(len(shrinkage_list))]
    beta_term_multiplier_true = [
        (shrinkage_list[i] + shrinkage_list[i] * np.trace(inverse[i] * psi_eigenvalues.T) / T) ** 2 for i in
        range(len(shrinkage_list))]

    return beta_term_multiplier_true, xi_beta_true

# needs some work for the variance

def dumb_w_matrix(features:np.ndarray,
                  shrinkage_list:np.ndarray)->np.ndarray:
    [T, P] = features.shape
    covariance = features.T @ features / T
    inverse = [np.linalg.pinv(z * np.eye(P) + covariance) for z in shrinkage_list]
    w_matrix = [features @ inverse[i] @ features.T / T for i in range(len(shrinkage_list))]
    return  w_matrix



def leave_one_out_dumb(labels: np.ndarray,
                       features: np.ndarray,
                       shrinkage_list: np.ndarray) -> float:
    """
    # Dumb implementation of Leave one out
    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param shrinkage_list:
    :return: Unbiased estimator
    """

    labels_squared = np.mean(labels ** 2)

    w_matrix = dumb_w_matrix(features = features,
    shrinkage_list=shrinkage_list)

    pi = compute_pi_t_tau(w_mat=w_matrix,
                          shrinkage_list=shrinkage_list,
                          labels=labels)

    # now, we compute R_{tau+1}(z) * pi_{times,tau} as a vector. The list is indexed by z while the vector is indexed by tau
    estimator_list = [labels * pi[i] for i in range(len(shrinkage_list))]

    estimator_perf = LeaveOut.estimator_performance(estimator_list, pi, labels_squared)
    return estimator_perf

def leave_one_out_estimator_performance(labels: np.ndarray,
                                        features: np.ndarray,
                                        eigenvalues: np.ndarray,
                                        eigenvectors: np.ndarray,
                                        shrinkage_list: np.ndarray) -> float:
    """
    # Lemma 30: Vectorized Leave one out
    # Implement leave one out estimator
    :param labels: Variables we wish to predict
    :param features: Signals we use to predict variables
    :param eigenvectors:
    :param eigenvalues:
    :param shrinkage_list:
    :return: Unbiased estimator
    """
    labels_squared = np.mean(labels ** 2)

    w_matrix = LeaveOut.smart_w_matrix(features=features,
                                       eigenvalues=eigenvalues,
                                       eigenvectors=eigenvectors,
                                       shrinkage_list=shrinkage_list)

    pi = compute_pi_t_tau(w_mat=w_matrix,
                          shrinkage_list=shrinkage_list,
                          labels=labels)

    # now, we compute R_{tau+1}(z) * pi_{times,tau} as a vector. The list is indexed by z while the vector is indexed by tau
    estimator_list = [labels * pi[i] for i in range(len(shrinkage_list))]

    estimator_perf = LeaveOut.estimator_performance(estimator_list, pi, labels_squared)

    return estimator_perf

def compute_pi_t_tau(w_mat,
                     shrinkage_list,
                     labels):

    one_over_one_minus_diag_of_w = [
        (1 / (1 - np.diag(w))).reshape(-1, 1) for w in w_mat]

    labels_normalized = [
        labels * (1 - n) for n in one_over_one_minus_diag_of_w]

    s_beta = [one_over_one_minus_diag_of_w[i] * (w_mat[i] @ labels) for i in
              range(len(shrinkage_list))]

    pi = [s_beta[i] + labels_normalized[i] for i in range(len(shrinkage_list))]
    return pi
