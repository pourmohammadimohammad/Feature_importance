from typing import List, Any

import numpy
import numpy as np
import pandas as pd

from main import *
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.marcenko_pastur import MarcenkoPastur
from parameters import *


# mohammad_is_wrong = RandomFeatures.naive_linear_single_underlying()

class LeaveOut:

    def __init__(self, t, c):
        self.times = None
        self.labels_oos_list = None
        self.estimator_out_of_sample_cumulative = None
        self.features_oos_list = None
        self.true_value_beta_eq_176 = None
        self.true_value_sigma_beta_eq_176 = None
        self.true_value_limit_eq_176 = None
        self.mean_true = None
        self.xi = None
        self.beta_eigenvalues = None
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
        self.c = c
        self.p = int(c * t)
        self.simple_beta = False

    def simulate_date(self, simple_beta=False):
        self.simple_beta = simple_beta
        labels, features, beta_dict, psi_eigenvalues, beta_eigenvalues = self.simulate_data(seed=self.seed,
                                                                                            sample_size=self.t,
                                                                                            number_features_=self.p,
                                                                                            beta_and_psi_link_=self.beta_and_psi_link,
                                                                                            noise_size_=self.noise_size,
                                                                                            activation_=self.activation,
                                                                                            number_neurons_=self.number_neurons,
                                                                                            simple_beta=self.simple_beta)
        self.labels = labels
        self.features = features
        self.beta_dict = beta_dict
        self.psi_eigenvalues = psi_eigenvalues
        self.beta_eigenvalues = beta_eigenvalues

    def train_test_split(self, train_frac):
        self.train_frac = train_frac
        split = int(self.t * train_frac)

        self.labels_ins = self.labels[:split]
        self.labels_oos = self.labels[split:]

        self.features_ins = self.features[:split, :]
        self.features_oos = self.features[split:, :]

    def test_parse_split(self, num_parts):
        oos_length = self.features_oos.shape[0]
        parts_frac = [0]
        parts_frac.extend(oos_length * np.linspace(1, num_parts, num_parts)/num_parts)
        oos_list_features = []
        oos_list_labels = []
        parts_frac = [int(parts_frac[i]) for i in range(len(parts_frac))]
        [oos_list_features.append(self.features_oos[parts_frac[i]:parts_frac[i + 1]]) for i in
         range(num_parts)]
        [oos_list_labels.append(self.labels_oos[int(parts_frac[i]):int(parts_frac[i + 1])]) for i in range(num_parts)]
        self.times = parts_frac[1:]
        self.features_oos_list = oos_list_features
        self.labels_oos_list = oos_list_labels

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

    def oos_performance(self, features_oos=None,labels_oos=None):
        features_oos = self.features_oos if features_oos is None else features_oos
        labels_oos = self.labels_oos if labels_oos is None else labels_oos

        self.estimator_oos = self.performance_oos(beta_hat=self.beta_hat,
                                                  labels_out_of_sample=labels_oos,
                                                  features_out_of_sample=features_oos)
        return self.estimator_oos

    def oos_performance_growing_sample(self, num_parts=4):
        self.test_parse_split(num_parts)
        estimator_out_of_sample = []
        [estimator_out_of_sample.append(LeaveOut.oos_performance(self, self.features_oos_list[i],self.labels_oos_list[i]))
         for i in range(len(self.features_oos_list))]
        estimator_out_of_sample_cumulative = {}
        for name in estimator_out_of_sample[0].keys():
            estimator_out_of_sample_cumulative[name] = []
            for i in range(len(self.features_oos_list)):
                estimator_out_of_sample_cumulative[name].append(np.array(estimator_out_of_sample[i][name]))
                if i != 0:
                    estimator_out_of_sample_cumulative[name][i] = estimator_out_of_sample_cumulative[name][i - 1] + \
                                                                  np.array(estimator_out_of_sample[i][name])
            estimator_out_of_sample_cumulative[name] = [estimator_out_of_sample_cumulative[name][i] / (i + 1)
                                                        for i in range(len(self.features_oos_list))]

        self.estimator_out_of_sample_cumulative = estimator_out_of_sample_cumulative
        return estimator_out_of_sample_cumulative

    def calculate_true_value(self):
        self.true_value_mean = LeaveOut.leave_one_out_true_value(beta_dict=self.beta_dict,
                                                                 psi_eigenvalues=self.psi_eigenvalues,
                                                                 eigenvalues=self.eigenvalues,
                                                                 eigenvectors=self.eigenvectors,
                                                                 shrinkage_list=self.shrinkage_list,
                                                                 noise_size_=self.noise_size)
        return self.true_value_mean

    def theoretical_mean(self):

        xi = LeaveOut.xi_beta_k_true(l=1,
                                     c=self.c,
                                     psi_eigenvalues=self.psi_eigenvalues,
                                     beta_eigenvalues=self.beta_eigenvalues,
                                     shrinkage_list=self.shrinkage_list)

        self.xi = xi
        mean_true = [np.sum(self.beta_eigenvalues * self.psi_eigenvalues) - self.shrinkage_list[i] * xi[i] for i in
                     range(len(self.shrinkage_list))]
        self.mean_true = mean_true

        return mean_true

    def True_value_eq_176(self, data_type):

        if data_type == DataUsed.INS:
            data = self.features_ins
        if data_type == DataUsed.OOS:
            data = self.features_oos
        if data_type == DataUsed.TOTAL:
            data = self.features

        [T, P] = data.shape
        covariance = data.T @ data / T
        inverse = [np.linalg.pinv(covariance + z * np.eye(P)) for z in self.shrinkage_list]
        true_value_sigma_beta_eq_176 = [np.trace(self.beta_eigenvalues * (self.psi_eigenvalues * i) @ covariance) for i
                                        in inverse]
        true_value_beta_eq_176 = [(self.beta_dict[0].reshape(1, -1) @ (self.psi_eigenvalues * i) @ covariance @
                                   self.beta_dict[0].reshape(-1, 1))[0] for i in inverse]
        self.true_value_sigma_beta_eq_176 = true_value_sigma_beta_eq_176
        self.true_value_beta_eq_176 = true_value_beta_eq_176

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

        w_diag = LeaveOut.smart_w_diag(features=features,
                                       eigenvalues=eigenvalues,
                                       eigenvectors=eigenvectors,
                                       shrinkage_list=shrinkage_list)

        pi = LeaveOut.compute_pi_t_tau_with_beta_hat(labels=labels,
                                                     features=features,
                                                     beta_hat=beta_hat,
                                                     w_diag=w_diag,
                                                     shrinkage_list=shrinkage_list)

        # now, we compute R_{tau+1}(z) * pi_{times,tau} as a vector. The list is indexed by z while the vector is indexed by tau
        estimator_list = [labels * pi[i] for i in range(len(shrinkage_list))]

        # Calculate strategy performance using insample dat
        estimator_perf = LeaveOut.estimator_performance(estimator_list, pi, labels_squared)

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

        return LeaveOut.estimator_performance(estimator_list, pi, labels_squared)

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
                      number_neurons_: int = 1,
                      simple_beta: bool = False
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
        np.random.seed(100)
        psi_eigenvalues = np.abs(np.random.uniform(0.01, 1, [1, number_features_]))

        # we should also experiment with non-monotonic links
        if simple_beta:
            beta_eigenvalues = np.ones([1, number_features_]) / number_features_
        else:
            beta_eigenvalues = psi_eigenvalues ** beta_and_psi_link_ / number_features_

        beta_dict = dict()
        for neuron in range(number_neurons_):
            betas = np.random.randn(number_features_, 1) * (beta_eigenvalues ** 0.5).reshape(-1, 1)
            beta_dict[neuron] = betas

        np.random.seed(seed)
        features = np.random.randn(sample_size, number_features_) * (psi_eigenvalues ** 0.5)

        labels_ = np.zeros([sample_size, 1])
        for neuron in range(number_neurons_):
            noise = np.random.randn(sample_size, 1) * noise_size_
            labels_ \
                += RandomFeaturesGenerator.apply_activation_to_multiplied_signals(
                multiplied_signals=features @ betas + noise,
                activation=activation_)

        return labels_, features, beta_dict, psi_eigenvalues, beta_eigenvalues

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

    @staticmethod
    def xi_beta_k_true(c: float,
                       l: int,
                       beta_eigenvalues: np.ndarray,
                       psi_eigenvalues: np.ndarray,
                       shrinkage_list: np.ndarray):
        """
        Efficient way to estimate \beta \Psi (\hat \Psi + zI)^{-1} \Psi \beta
        :param beta_dict: beta paramaters (ground truth)
        :param psi_eigenvalues: True eigenvalues of covariance matrix
        :param shrinkage_list:
        :param T: Sample size
        :return:
        """

        # m_z_c = MarcenkoPastur.marcenko_pastur(c,shrinkage_list)
        # denominator = 1/(1-c+c*shrinkage_list*m_z_c)
        # params = [ denominator[i]*shrinkage_list[i] for i in range(len(shrinkage_list))]
        #
        # inverse_eigen_values = [(psi_eigenvalues ** l)/(psi_eigenvalues+p) for p  in params]
        #
        # xi = [np.trace(beta_eigenvalues*inverse_eigen_values[i])*denominator[i] for i in range(len(shrinkage_list))]

        m_z_c = MarcenkoPastur.marcenko_pastur(c, shrinkage_list)
        denominator = 1 / (1 - c + c * shrinkage_list * m_z_c)
        params = [denominator[i] * shrinkage_list[i] for i in range(len(shrinkage_list))]

        inverse_eigen_values = [(psi_eigenvalues ** l) / (psi_eigenvalues + p) for p in params]

        xi = [np.sum(beta_eigenvalues * inverse_eigen_values[i]) * denominator[i] for i in
              range(len(shrinkage_list))]

        return xi
