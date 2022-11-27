import numpy
import numpy as np
import pandas as pd

from main import *
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.marcenko_pastur import MarcenkoPastur


# mohammad_is_wrong = RandomFeatures.naive_linear_single_underlying()

class leave_out:
    def __init__(self, t, c):
        self.true_value_mean = None
        self.true_value_var = None
        self.estimator_oos = None
        self.performance_ins = None
        self.pi_avg = None
        self.beta_hat = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.features_oos = None
        self.features_ins = None
        self.labels_oos = None
        self.labels_ins = None
        self.train_frac = None
        self.psi_eigenvalues = None
        self.beta_dict = None
        self.features = None
        self.labels = None
        self.seed = 0
        self.beta_and_psi_link = 2
        self.noise_size = 0
        self.activation = 'linear'
        self.number_neurons = 1
        self.shrinkage_list = np.linspace(0.1, 10, 100)
        self.t = t
        self.p = int(c * t)

    def simulate_date(self):
        labels, features, beta_dict, psi_eigenvalues = self.simulate_data(seed=self.seed,
                                                                          sample_size=self.t,
                                                                          number_features_=self.p,
                                                                          beta_and_psi_link_=self.beta_and_psi_link,
                                                                          noise_size_=self.noise_size,
                                                                          activation_=self.activation,
                                                                          number_neurons_=self.number_neurons)
        self.labels = labels
        self.features = features
        self.beta_dict = beta_dict
        self.psi_eigenvalues = psi_eigenvalues

    def train_test_split(self, train_frac):
        self.train_frac = train_frac
        split = int(self.t * train_frac)

        self.labels_ins = self.labels[:split]
        self.labels_oos = self.labels[split:]

        self.features_ins = self.features[:split, :]
        self.features_oos = self.features[split:, :]


    def train_model(self):
        self.eigenvalues, self.eigenvectors = self.smart_eigenvalue_decomposition(self.features_ins)

        self.beta_hat = self.smart_beta_hat(labels=self.labels_ins,
                                            features=self.features_ins,
                                            eigenvalues=self.eigenvalues,
                                            eigenvectors=self.eigenvectors,
                                            shrinkage_list=self.shrinkage_list)

    def ins_performance(self):
        self.performance_ins, self.pi_avg = self.leave_one_out_estimator_beta(labels=self.labels_ins,
                                                                              features=self.features_ins,
                                                                              eigenvalues=self.eigenvalues,
                                                                              eigenvectors=self.eigenvectors,
                                                                              beta_hat=self.beta_hat,
                                                                              shrinkage_list=self.shrinkage_list)
        return self.performance_ins

    def oos_performance(self):
        self.estimator_oos = self.performance_oos(beta_hat=self.beta_hat,
                                                  labels_out_of_sample=self.labels_oos,
                                                  features_out_of_sample=self.features_oos)
        return self.estimator_oos

    def calculate_true_value(self):
        self.true_value_mean = leave_out.leave_one_out_true_value(beta_dict=self.beta_dict,
                                                                                       psi_eigenvalues=self.psi_eigenvalues,
                                                                                       eigenvalues=self.eigenvalues,
                                                                                       eigenvectors=self.eigenvectors,
                                                                                       shrinkage_list=self.shrinkage_list,
                                                                                       noise_size_=self.noise_size)
        return self.true_value_mean

    @staticmethod
    def smart_eigenvalue_decomposition(features: np.ndarray,
                                       T: int = None):
        """
        Lemma 28: Efficient Eigen value decomposition
        :param features: features used to create covariance matrix times x P
        :param T: Weight used to normalize matrix
        :return: Left eigenvectors PxT and eigenvalues without zeros
        """
        [T_true, P] = features.shape
        T = T_true if T is None else T

        if P > T:
            covariance = features @ features.T / T

        else:
            covariance = features.T @ features / T

        eigval, eigvec = np.linalg.eigh(covariance)
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]

        if P > T:
            # project features on normalized eigenvectors
            eigvec = np.matmul(features.T, eigvec * ((eigval * T) ** (-1 / 2)).reshape(1, -1))

        return eigval, eigvec

    @staticmethod
    def smart_w_matrix(features: np.ndarray,
                       eigenvalues: np.ndarray,
                       eigenvectors: np.ndarray,
                       shrinkage_list: np.ndarray):
        """
         Smart W calculation
        (z+Psi)^{-1} = U (z+lambda)^{-1}U' + z^{-1} (I - UU')
        we compute S'(z+Psi)^{-1} S= S' (
        :param features:
        :param eigenvalues:
        :param eigenvectors:
        :param shrinkage_list:
        :return:
        """
        [T, P] = features.shape
        # we project the features on the eigenvectors
        projected_features = eigenvectors.T @ features.T / np.sqrt(T)
        # we normalized the projected features by eigen values
        stuff_divided = [(1 / (eigenvalues.reshape(-1, 1) + z)) * projected_features for z in shrinkage_list]

        W = [projected_features.times @ x_ for x_ in stuff_divided]

        if P > T:
            # adjustment for zero eigen values
            cov_left_over = features @ features.T / T - projected_features.times @ projected_features
            W = [W[i] + (1 / shrinkage_list[i]) * cov_left_over for i in range(len(shrinkage_list))]

        return W

    @staticmethod
    def smart_w_diag(features: np.ndarray,
                     eigenvalues: np.ndarray,
                     eigenvectors: np.ndarray,
                     shrinkage_list: np.ndarray):
        """
        Smart calculation for the diagonal of W
        :param features:
        :param eigenvalues:
        :param eigenvectors:
        :param shrinkage_list:
        :return:
        """
        [T, P] = features.shape
        # we project the features on the eigenvectors
        projected_features = eigenvectors.T @ features.T / np.sqrt(T)
        # we normalized the projected features by eigen values
        stuff_divided = [(1 / (eigenvalues.reshape(-1, 1) + z)) * projected_features for z in shrinkage_list]

        W = [np.sum((projected_features * x_), axis=0) for x_ in stuff_divided]

        if P > T:
            # adjustment for zero eigen values
            cov_left_over = (features.T * features.T).sum(0) / T - (projected_features * projected_features).sum(0)
            W = [W[i] + (1 / shrinkage_list[i]) * cov_left_over for i in range(len(shrinkage_list))]

        return W

    @staticmethod
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
        beta_hat = [
            eigenvectors @ ((1 / (eigenvalues.reshape(-1, 1) + z)) * projected_features) @ labels.reshape(-1, 1) / T
            for z in shrinkage_list]

        if P > T:
            beta_hat_adj = features.T @ labels.reshape(-1, 1) - eigenvectors @ projected_features @ labels.reshape(-1,
                                                                                                                   1)
            beta_hat = [beta_hat[i] + beta_hat_adj / shrinkage_list[i] for i in range(len(shrinkage_list))]

        return beta_hat

    @staticmethod
    def leave_one_out_estimator_beta(labels: np.ndarray,
                                     features: np.ndarray,
                                     eigenvalues: np.ndarray,
                                     eigenvectors: np.ndarray,
                                     beta_hat: np.ndarray,
                                     shrinkage_list: np.ndarray) -> float:
        """
        # Lemma 30: Vectorized Leave one out
        # Implement leave one out estimator
        :param labels: Variables we wish to predict
        :param features: Signals we use to predict variables
        :param eigenvectors: calculated form in-sample data
        :param eigenvalues:calculated form in-sample data
        :param shrinkage_list:
        :return: Unbiased estimator
        """
        labels_squared = np.mean(labels ** 2)

        w_diag = leave_out.smart_w_diag(features=features,
                                        eigenvalues=eigenvalues,
                                        eigenvectors=eigenvectors,
                                        shrinkage_list=shrinkage_list)

        pi = leave_out.compute_pi_t_tau_with_beta_hat(labels=labels,
                                                      features=features,
                                                      beta_hat=beta_hat,
                                                      w_diag=w_diag,
                                                      shrinkage_list=shrinkage_list)

        # now, we compute R_{tau+1}(z) * pi_{times,tau} as a vector. The list is indexed by z while the vector is indexed by tau
        estimator_list = [labels * pi[i] for i in range(len(shrinkage_list))]

        # Calculate strategy performance using insample dat
        estimator_perf = leave_out.estimator_performance(estimator_list, pi, labels_squared)

        # do an average over all pi_T_tau do get the \hat \pi estimator
        pi_avg = [np.mean(p) for p in pi]

        return estimator_perf, pi_avg

    @staticmethod
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

    @staticmethod
    def performance_oos(beta_hat: np.ndarray,
                        labels_out_of_sample: np.ndarray,
                        features_out_of_sample: np.ndarray):
        """
        Use beta estimate to test out of sample performance
        :param beta_hat:
        :param labels_out_of_sample:
        :param features_out_of_sample:
        :return: performance
        """
        t = features_out_of_sample.shape[0]
        labels_squared = np.mean(labels_out_of_sample ** 2)
        pi = [features_out_of_sample @ b for b in beta_hat]
        estimator_list = [p * labels_out_of_sample for p in pi]

        return leave_out.estimator_performance(estimator_list, pi, labels_squared)

    @staticmethod
    def compute_pi_t_tau_with_beta_hat(labels,
                                       features,
                                       w_diag,
                                       beta_hat,
                                       shrinkage_list):

        one_over_one_minus_diag_of_w = [
            (1 / (1 - w)).reshape(-1, 1) for w in w_diag]

        labels_normalized = [
            labels * (1 - n) for n in one_over_one_minus_diag_of_w]

        s_beta = [one_over_one_minus_diag_of_w[i] * (features @ beta_hat[i]) for i in
                  range(len(shrinkage_list))]

        pi = [s_beta[i] + labels_normalized[i] for i in range(len(shrinkage_list))]
        return pi

    @staticmethod
    def estimator_performance(estimator_list, pi, labels_squared):
        estimator_list_mean = [np.mean(e) for e in estimator_list]

        estimator_list_std = [np.std(e) for e in estimator_list]

        estimator_list_pi_2 = [np.mean(p ** 2) for p in pi]

        estimator_list_pi = [np.mean(p) for p in pi]

        estimator_list_sharpe = [estimator_list_mean[i] / estimator_list_std[i]
                                 for i in range(len(estimator_list))]

        estimator_list_mse = [labels_squared - 2 * estimator_list_mean[i] + estimator_list_pi_2[i]
                              for i in range(len(estimator_list))]
        estimator_perf = {}
        estimator_perf['mean'] = estimator_list_mean
        estimator_perf['std'] = estimator_list_std
        estimator_perf['pi_2'] = estimator_list_pi_2
        estimator_perf['mse'] = estimator_list_mse
        estimator_perf['sharpe'] = estimator_list_sharpe
        estimator_perf['pi'] = estimator_list_pi
        return estimator_perf

    @staticmethod
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

        beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_ / np.sqrt(number_features_)

        :param noise_size_: size of noise
        :return: labels and features. Labels are noisy and potentially non-linear functions of features
        """
        np.random.seed(seed)
        psi_eigenvalues = np.abs(np.random.uniform(0.01, 1, [1, number_features_]))
        features = np.random.randn(sample_size, number_features_) * (psi_eigenvalues ** 0.5)

        # should also divide by P to ensure bounded trace norm
        beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_ / np.sqrt(number_features_)
        # we should also experiment with non-monotonic links

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

    @staticmethod
    def leave_one_out_true_value(beta_dict: np.ndarray,
                                 psi_eigenvalues: np.ndarray,
                                 eigenvalues: np.ndarray,
                                 eigenvectors: np.ndarray,
                                 shrinkage_list: np.ndarray,
                                 noise_size_: float):
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
        # T = eigenvectors.shape[1]
        psi_beta = psi_eigenvalues.reshape(-1, 1) * beta_dict[0]
        beta_psi_beta = np.sum(psi_beta * beta_dict[0])
        eigenvectors_projection_psi_beta = eigenvectors.T @ psi_beta
        eigenvectors_projection_beta = eigenvectors.T @ beta_dict[0]
        eigenvalues = eigenvalues.reshape(1, -1)

        left_over_psi_beta = beta_psi_beta - np.sum(eigenvectors_projection_psi_beta * eigenvectors_projection_beta)

        xi_psi_beta = [
            ((1 / (eigenvalues + z)) @ (eigenvectors_projection_psi_beta * eigenvectors_projection_beta))[0]
            for z in
            shrinkage_list]

        xi_psi_beta_complete = [shrinkage_list[i] * xi_psi_beta[i] + left_over_psi_beta for i in
                                range(len(shrinkage_list))]

        true_values_mean = [(beta_psi_beta - xi_psi_beta_complete[i])[0]
                            for i in range(len(shrinkage_list))]

        # eigenvectors_projection_psi = eigenvectors.T * psi_eigenvalues

        # xi = [
        #     np.trace((eigenvectors * (1 / (eigenvalues + z)) @ eigenvectors_projection_psi)) / T
        #     for z in
        #     shrinkage_list]
        # left_over_xi = (np.sum(psi_eigenvalues) - np.trace(eigenvectors @ eigenvectors_projection_psi)) / T

        # xi_der = [
        #     np.trace(eigenvectors * (1 / (eigenvalues + z) ** 2) @ eigenvectors_projection_psi) / T
        #     for z in
        #     shrinkage_list]

        # no_beta_term = [
        #     (noise_size_ ** 2) * (xi[i] - shrinkage_list[i] * xi_der[i]) for i in
        #     range(len(shrinkage_list))]
        #
        # xi_beta = [
        #     ((eigenvalues / ((eigenvalues + z) ** 2)) @ (eigenvectors_projection_beta ** 2))[0]
        #     for z in shrinkage_list]

        # leftovers for the derivatives cancel each other out

        # beta_term_multiplier = [(shrinkage_list[i] + shrinkage_list[i] * xi[i] + left_over_xi) ** 2
        #                         for i in range(len(shrinkage_list))]

        # beta_term = [beta_term_multiplier[i] * xi_beta[i] for i in range(len(shrinkage_list))]

        # true_values_std = [(beta_psi_beta + noise_size_ ** 2) *
        #                    (true_values_mean[i] - xi_psi_beta_complete[i]
        #                     + beta_term[i] + no_beta_term[i]) - true_values_mean[i] ** 2
        #                    for i in range(len(shrinkage_list))]

        # true_values_std = [np.sqrt(t) for t in true_values_std]

        return true_values_mean
