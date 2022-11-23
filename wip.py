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

# Todo: Fix variance estimate
# Todo: leave one out and leave two out equivalane
# Todo: adjustment should only happen when you are in complex land
# Todo: Bring everything on server

def smart_eigenvalue_decomposition(features: np.ndarray,
                                   T: int = None):
    """
    Lemma 28: Efficient Eigen value decomposition
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
        # project features on normalized eigenvectors
        eigvec = np.matmul(features.T, eigvec * ((eigval * T) ** (-1 / 2)).reshape(1, -1))

    return eigval, eigvec


def smart_w_matrix(features: np.ndarray,
                   eigenvalues: np.ndarray,
                   eigenvectors: np.ndarray,
                   shrinkage_list: np.ndarray):
    """
    Lemma 29: Smart W calculation
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

    if P > T:
        cov_left_over = features @ features.T / T - projected_features.T @ projected_features
        W = [W[i] + (1 / shrinkage_list[i]) * cov_left_over for i in range(len(shrinkage_list))]

    return W


def smart_beta_hat(labels: np.ndarray,
                   features: np.ndarray,
                   eigenvalues: np.ndarray,
                   eigenvectors: np.ndarray,
                   shrinkage_list: np.ndarray):
    """
    (z+Psi)^{-1} = U (z+lambda)^{-1}U' + z^{-1} (I - UU')
    we compute S'(z+Psi)^{-1} S= S' (
    :param labels:
    :param features:
    :param eigenvalues:
    :param eigenvectors:
    :param shrinkage_list:
    :return:
    """
    [T, P] = features.shape
    projected_features = eigenvectors.T @ features.T
    beta_hat = [eigenvectors @ ((1 / (eigenvalues.reshape(-1, 1) + z)) * projected_features) @ labels.reshape(-1, 1) / T
                for z in shrinkage_list]

    if P > T:
        beta_hat_adj = labels.reshape(-1, 1) - eigenvectors @ projected_features @ labels.reshape(-1, 1)
        beta_hat = [beta_hat[i] + beta_hat_adj / shrinkage_list[i] for i in range(len(shrinkage_list))]

    return beta_hat


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

    num = (T - 1)  # divided by T to account for W normalization
    labels = labels.reshape(-1, 1)

    estimator_list = [(labels.T @ map_w_to_w_tilde(w_matrix) @ labels / num)[0] for w_matrix in W]

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

    num = (T - 1)  # divded by T to account for W normalization

    matrix_multiplied = features.T @ A @ features
    np.fill_diagonal(matrix_multiplied, 0)

    labels = labels.reshape(-1, 1)

    estimator = (labels.T @ matrix_multiplied @ labels / num)[0]

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
    m_list = empirical_stieltjes(eigenvalues, P, shrinkage_list)
    mean_estimator = [(1 - c + c * shrinkage_list[i] * m_list[i]) * estimator_list[i] for i in
                      range(len(shrinkage_list))]

    return mean_estimator


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

    W = smart_w_matrix(features=features,
                       eigenvalues=eigenvalues,
                       eigenvectors=eigenvectors,
                       shrinkage_list=shrinkage_list)

    normalizer = [
        (1 / (1 - np.diag(w))).reshape(-1, 1) for w in W]

    labels_normalized = [
        labels * (1 - n) for n in normalizer]

    s_beta = [normalizer[i] * (W[i] @ labels) for i in
              range(len(shrinkage_list))]

    pi = [s_beta[i] + labels_normalized[i] for i in range(len(shrinkage_list))]
    estimator_list = [labels * pi[i] for i in range(len(shrinkage_list))]
    estimator_list_mean = [np.mean(e) for e in estimator_list]
    estimator_list_std = [np.std(e) for e in estimator_list]
    estimator_list_pi_2 = [np.mean(p ** 2) for p in pi]
    estimator_list_sharpe = [estimator_list_mean[i] / estimator_list_std[i]
                             for i in range(len(shrinkage_list))]
    labels_squared = np.mean(labels ** 2)
    estimator_list_mse = [labels_squared - 2 * estimator_list_mean[i] + estimator_list_pi_2[i]
                          for i in range(len(shrinkage_list))]
    estimators = {}
    estimators['mean'] = estimator_list_mean
    estimators['std'] = estimator_list_std
    estimators['pi_2'] = estimator_list_pi_2
    estimators['mse'] = estimator_list_mse
    estimators['sharpe'] = estimator_list_sharpe

    return estimators


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
    [T,P] = features.shape
    covariance = features.T @ features /T
    inverse = [ np.linalg.pinv(z*np.eye(P) + covariance) for z in shrinkage_list]
    W = [features@inverse[i]@features.T/T for i in range(len(shrinkage_list))]
    normalizer = [
        (1 / (1 - np.diag(w))).reshape(-1, 1) for w in W]

    labels_normalized = [
        labels * (1 - n) for n in normalizer]

    s_beta = [normalizer[i] * (W[i]@labels) for i in
              range(len(shrinkage_list))]

    pi = [s_beta[i] + labels_normalized[i] for i in range(len(shrinkage_list))]
    estimator_list = [labels * pi[i] for i in range(len(shrinkage_list))]
    estimator_list_mean = [np.mean(e) for e in estimator_list]
    estimator_list_std = [np.std(e) for e in estimator_list]
    return estimator_list_mean, estimator_list_std


def leave_one_out_true_value(beta_dict: np.ndarray,
                             psi_eigenvalues: np.ndarray,
                             eigenvalues: np.ndarray,
                             eigenvectors: np.ndarray,
                             shrinkage_list: np.ndarray,
                             noise_size_: float,
                             T: int):
    """
    Efficient way to estimate \beta \Psi (\hat \Psi + zI)^{-1} \Psi \beta
    :param beta_dict: beta paramaters (ground truth)
    :param psi_eigenvalues: True eigenvalues of covariance matrix
    :param eigenvalues: eigenvalues of covariance matrix
    :param eigenvectors: eigenvectors of covariance matrix
    :param shrinkage_list:
    :param noise_size_: Size of noise standard deviation
    :param T: Sample size
    :return:
    """

    psi_beta = psi_eigenvalues.reshape(-1, 1) * beta_dict[0]
    beta_psi_beta = np.sum(psi_beta * beta_dict[0])
    eigenvalues, eigenvectors = smart_eigenvalue_decomposition(features)

    eigenvectors_projection_psi_beta = eigenvectors.T @ psi_beta
    eigenvectors_projection_beta = eigenvectors.T @ beta_dict[0]
    eigenvectors_projection_psi = eigenvectors.T * psi_eigenvalues
    eigenvalues = eigenvalues.reshape(1, -1)

    xi = [
        np.trace((eigenvectors * (1 / (eigenvalues + z)) @ eigenvectors_projection_psi)) / T
        for z in
        shrinkage_list]
    left_over_xi = (np.sum(psi_eigenvalues) - np.trace(eigenvectors @ eigenvectors_projection_psi)) / T

    xi_der = [
        np.trace(eigenvectors * (1 / (eigenvalues + z) ** 2) @ eigenvectors_projection_psi) / T
        for z in
        shrinkage_list]

    no_beta_term = [
        (noise_size_ ** 2) * (xi[i] - shrinkage_list[i] * xi_der[i]) for i in
        range(len(shrinkage_list))]

    xi_beta = [
        ((eigenvalues / ((eigenvalues + z) ** 2)) @ (eigenvectors_projection_beta ** 2))[0]
        for z in shrinkage_list]

    # leftovers for the derivatives cancel each other out

    beta_term_multiplier = [(shrinkage_list[i] + shrinkage_list[i] * xi[i] + left_over_xi) ** 2
                            for i in range(len(shrinkage_list))]

    beta_term = [beta_term_multiplier[i] * xi_beta[i] for i in range(len(shrinkage_list))]

    left_over_psi_beta = beta_psi_beta - np.sum(eigenvectors_projection_psi_beta * eigenvectors_projection_beta)

    xi_psi_beta = [
        ((1 / (eigenvalues + z)) @ (eigenvectors_projection_psi_beta * eigenvectors_projection_beta))[0]
        for z in
        shrinkage_list]

    xi_psi_beta_complete = [shrinkage_list[i] * xi_psi_beta[i] + left_over_psi_beta for i in range(len(shrinkage_list))]

    true_values_mean = [beta_psi_beta - xi_psi_beta_complete[i]
                        for i in range(len(shrinkage_list))]

    true_values_std = [(beta_psi_beta + noise_size_ ** 2) *
                       (true_values_mean[i] - xi_psi_beta_complete[i]
                        + beta_term[i] + no_beta_term[i]) - true_values_mean[i] ** 2
                       for i in range(len(shrinkage_list))]

    true_values_std = [np.sqrt(t) for t in true_values_std]

    return true_values_mean, true_values_std


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


if __name__ == '__main__':
    # testing leave one out:
    name_list = ['mean', 'std']
    T = [20, 100, 500,1000]
    # C = [0.1, 0.5, 1, 2, 5, 10]
    C = [0.5, 2]

    fig_dict = {}
    axs_dict = {}
    error_dict = {}

    seed = 0
    beta_and_psi_link_ = 2
    noise_size_ = 0
    activation_ = 'linear'
    number_neurons_ = 1
    shrinkage_list = np.linspace(0.1, 1, 100)

    # c = 0.5
    # sample_size = 100
    # number_features_ = int(c * sample_size)
    # split = int(sample_size / 2)
    #
    # labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
    #                                                              sample_size=sample_size,
    #                                                              number_features_=number_features_,
    #                                                              beta_and_psi_link_=beta_and_psi_link_,
    #                                                              noise_size_=noise_size_,
    #                                                              activation_=activation_,
    #                                                              number_neurons_=number_neurons_)
    #
    # eigenvalues, eigenvectors = smart_eigenvalue_decomposition(features)
    #
    # estimator = leave_one_out_estimator_performance(
    #             labels,
    #             features,
    #             eigenvalues,
    #             eigenvectors,
    #             shrinkage_list)
    #
    # estimator_dumb=leave_one_out_dumb(labels, features,shrinkage_list)
    #
    # plt.plot(shrinkage_list, estimator['mean'])
    # plt.plot(shrinkage_list, estimator_dumb[0] )
    # plt.legend(['estimate','dumb'])
    #
    # plt.show()



    estmator_dict = {}

    for name in name_list:
        estmator_dict[name] = pd.DataFrame(index=T, columns=C)


    ax_legend = []
    for c in C:
        ax_legend.append(f'Complexity = {c} INS')
        ax_legend.append(f'Complexity = {c} OOS')

    for i in range(len(T)):

        for j in range(len(C)):
            sample_size = T[i]
            number_features_ = int(C[j] * sample_size)
            split = int(sample_size / 2)

            labels, features, beta_dict, psi_eigenvalues = simulate_data(seed=seed,
                                                                         sample_size=sample_size,
                                                                         number_features_=number_features_,
                                                                         beta_and_psi_link_=beta_and_psi_link_,
                                                                         noise_size_=noise_size_,
                                                                         activation_=activation_,
                                                                         number_neurons_=number_neurons_)

            labels_in_sample = labels[:split]
            labels_out_of_sample = labels[split:]

            features_in_sample = features[:split, :]
            features_out_of_sample = features[split:, :]

            eigenvalues_in_sample, eigenvectors_in_sample = smart_eigenvalue_decomposition(features_in_sample)
            eigenvalues_out_of_sample, eigenvectors_out_of_sample = smart_eigenvalue_decomposition(
                features_out_of_sample)

            estimator_in_sample = leave_one_out_estimator_performance(
                labels_in_sample,
                features_in_sample,
                eigenvalues_in_sample,
                eigenvectors_in_sample,
                shrinkage_list)
            estimator_out_of_sample = leave_one_out_estimator_performance(
                labels_out_of_sample, features_out_of_sample,
                eigenvalues_out_of_sample,
                eigenvectors_out_of_sample,
                shrinkage_list)

            for name in name_list:
                estmator_dict[name].loc[T[i],C[j]] = [{'INS':estimator_in_sample[name],'OOS':estimator_out_of_sample[name]}]


    error_dict = {}
    for name in name_list:
        fig, ax = plt.subplots(2, 2, sharex=True, figsize=(15, 15))
        error_dict[name]={}
        for i in range(len(T)):
            a_0 = i % 2
            a_1 = int((i - a_0) / 2) % 2
            for j in range(len(C)):
                ins = estmator_dict[name].loc[T[i],C[j]][0]['INS']
                oos = estmator_dict[name].loc[T[i],C[j]][0]['OOS']
                ax[a_0, a_1].plot(shrinkage_list, ins)
                ax[a_0, a_1].plot(shrinkage_list,oos)
                ax[a_0, a_1].set_title(f'T = {T[i]}')
                error_dict[name][T[i]] = np.mean(np.abs(np.array(ins) - \
                                                        np.array(oos)) / np.array(oos))
        ax[0, 0].legend(ax_legend, loc='upper right')
        fig.text(0.5, 0.04, 'z shrinkage', ha='center')
        fig.suptitle(name + f" with beta_and_psi_link_ = {beta_and_psi_link_}", fontsize=14)
        plt.show()



