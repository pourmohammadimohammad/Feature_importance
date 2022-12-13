from typing import List, Any

import numpy
import numpy as np
import pandas as pd

from main import *
from rf.RandomFeaturesGenerator import RandomFeaturesGenerator
from helpers.random_features import RandomFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm # use for showing progress in loops
from helpers.marcenko_pastur import MarcenkoPastur
from parameters import *


# mohammad_is_wrong = RandomFeatures.naive_linear_single_underlying()

class LeaveOut:

    def __init__(self, par:Params = None, labels:np.ndarray = None, features:np.ndarray = None):
        self.par = par
        self.oos_m_mse = None
        self.oos_m_sharpe = None
        self.oos_r_z = None
        self.oos_pi_z = None
        self.ins_m_mse = None
        self.ins_m_sharpe = None
        self.m_z_c = None
        self.var_true = None
        self.xi_1 = None
        self.oos_optimal_sharpe = None
        self.oos_optimal_mse = None
        self.beta_hat_optimal_mse = None
        self.beta_hat_optimal_sharpe = None
        self.pi_ins = None
        self.ret_vec_ins = None
        self.times = None
        self.labels_oos_list = None
        self.estimator_out_of_sample_cumulative = None
        self.features_oos_list = None
        self.true_value_beta_eq_176 = None
        self.true_value_sigma_beta_eq_176 = None
        self.true_value_limit_eq_176 = None
        self.mean_true = None
        self.xi_beta_1 = None
        self.beta_eigenvalues = None
        self.true_value_mean = None
        self.true_value_var = None
        self.oos_perf_est = None
        self.ins_perf_est = None
        self.pi_avg = None
        self.beta_hat = None
        self.eigenvectors = None
        self.eigenvalues = None
        self.features_oos = None
        self.features_ins = None
        self.labels_oos = None
        self.labels_ins = None
        self.psi_eigenvalues = None
        self.beta_dict = None
        self.features = features
        self.labels = labels

    def save(self, save_dir, file_name='/leave_out.p'):
        # simple save function that allows loading of deprecated parameters object
        df = pd.DataFrame(columns=['key', 'value'])

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                    df = df.append(temp)

            except:
                temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                df = df.append(temp)
        df.to_pickle(save_dir + file_name, protocol=4)

    def load(self, load_dir, file_name='/leave_out.p'):
        # simple load function that allows loading of deprecated parameters object
        df = pd.read_pickle(load_dir + file_name)
        # First check if this is an old pickle version, if so transform it into a df
        if type(df) != pd.DataFrame:
            loaded_par = df
            df = pd.DataFrame(columns=['key', 'value'])
            for key, v in loaded_par.__dict__.items():
                try:
                    for key2, vv in v.__dict__.items():
                        temp = pd.DataFrame(data=[str(key) + '_' + str(key2), vv], index=['key', 'value']).T
                        df = df.append(temp)

                except:
                    temp = pd.DataFrame(data=[key, v], index=['key', 'value']).T
                    df = df.append(temp)

        no_old_version_bug = True

        for key, v in self.__dict__.items():
            try:
                for key2, vv in v.__dict__.items():
                    t = df.loc[df['key'] == str(key) + '_' + str(key2), 'value']
                    if t.shape[0] == 1:
                        tt = t.values[0]
                        self.__dict__[key].__dict__[key2] = tt
                    else:
                        if no_old_version_bug:
                            no_old_version_bug = False
                            print('#### Loaded parameters object is depreceated, default version will be used')
                        print('Parameter', str(key) + '.' + str(key2), 'not found, using default: ',
                              self.__dict__[key].__dict__[key2])

            except:
                t = df.loc[df['key'] == str(key), 'value']
                if t.shape[0] == 1:
                    tt = t.values[0]
                    self.__dict__[key] = tt
                else:
                    if no_old_version_bug:
                        no_old_version_bug = False
                        print('#### Loaded parameters object is depreceated, default version will be used')
                    print('Parameter', str(key), 'not found, using default: ', self.__dict__[key])

    def train_test_split(self):
        split = int(self.par.simulated_data.t * self.par.plo.train_frac)

        self.labels_ins = self.labels[:split]
        self.labels_oos = self.labels[split:]

        self.features_ins = self.features[:split, :]
        self.features_oos = self.features[split:, :]

    def test_parse_split(self, num_parts):
        oos_length = self.features_oos.shape[0]
        parts_frac = [0]
        parts_frac.extend(oos_length * np.linspace(1, num_parts, num_parts) / num_parts)
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
                                            shrinkage_list=self.par.plo.shrinkage_list)

    def ins_performance(self):
        self.ins_perf_est, self.ret_vec_ins, self.pi_ins, self.pi_avg = self.leave_one_out_estimator_beta(
            labels=self.labels_ins,
            features=self.features_ins,
            eigenvalues=self.eigenvalues,
            eigenvectors=self.eigenvectors,
            beta_hat=self.beta_hat,
            shrinkage_list=self.par.plo.shrinkage_list)

        w_sharpe, w_mse, self.ins_m_sharpe, self.ins_m_mse = self.optimal_weights(self.ret_vec_ins,
                                                                                  self.pi_ins)

        self.beta_hat_optimal_sharpe, self.beta_hat_optimal_mse, = self.optimal_shrinkage(w_sharpe=w_sharpe,
                                                                                          w_mse=w_mse,
                                                                                          beta_hat=self.beta_hat)

    def oos_performance(self, features_oos=None, labels_oos=None):
        features_oos = self.features_oos if features_oos is None else features_oos
        labels_oos = self.labels_oos if labels_oos is None else labels_oos

        self.oos_perf_est, self.oos_pi_z, self.oos_r_z = self.performance_oos(beta_hat=self.beta_hat,
                                                                              labels_out_of_sample=labels_oos,
                                                                              features_out_of_sample=features_oos)
        
        w_sharpe, w_mse, self.oos_m_sharpe, self.oos_m_mse = self.optimal_weights(self.oos_r_z,
                                                                                  self.oos_pi_z)

        self.oos_optimal_sharpe = self.performance_oos(beta_hat=self.beta_hat_optimal_sharpe,
                                                       labels_out_of_sample=labels_oos,
                                                       features_out_of_sample=features_oos)[0]['sharpe']

        self.oos_optimal_mse = self.performance_oos(beta_hat=self.beta_hat_optimal_mse,
                                                    labels_out_of_sample=labels_oos,
                                                    features_out_of_sample=features_oos)[0]['mse']

    def oos_performance_growing_sample(self, num_parts=4):
        self.test_parse_split(num_parts)
        estimator_out_of_sample = []
        [estimator_out_of_sample.append(
            LeaveOut.oos_performance(self, self.features_oos_list[i], self.labels_oos_list[i]))
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

    def calculate_true_value(self):
        self.true_value_mean = LeaveOut.leave_one_out_true_value(beta_dict=self.beta_dict,
                                                                 psi_eigenvalues=self.psi_eigenvalues,
                                                                 eigenvalues=self.eigenvalues,
                                                                 eigenvectors=self.eigenvectors,
                                                                 shrinkage_list=self.par.plo.shrinkage_list,
                                                                 noise_size_=self.par.simulated_data.noise_size)

    def theoretical_mean_var(self):

        c = self.par.simulated_data.c / self.train_frac

        m_z_c = self.empirical_stieltjes(self.eigenvalues, self.par.simulated_data.p, self.par.plo.shrinkage_list)

        self.m_z_c = m_z_c

        xi_beta_1 = self.xi_k_true(c=c,
                                   psi_eigenvalues=self.psi_eigenvalues,
                                   beta_eigenvalues=self.beta_eigenvalues,
                                   shrinkage_list=self.par.plo.shrinkage_list,
                                   m_z_c=m_z_c)

        xi_beta_0 = self.xi_k_true(l=0,
                                   c=c,
                                   psi_eigenvalues=self.psi_eigenvalues,
                                   beta_eigenvalues=self.beta_eigenvalues,
                                   shrinkage_list=self.par.plo.shrinkage_list,
                                   m_z_c=m_z_c)

        derivative_xi_beta_0 = self.xi_k_true(l=0,
                                              c=c,
                                              d=2,
                                              psi_eigenvalues=self.psi_eigenvalues,
                                              beta_eigenvalues=self.beta_eigenvalues,
                                              shrinkage_list=self.par.plo.shrinkage_list,
                                              m_z_c=m_z_c)

        xi_1 = self.xi_k_true(c=c,
                              psi_eigenvalues=self.psi_eigenvalues,
                              shrinkage_list=self.par.plo.shrinkage_list,
                              m_z_c=m_z_c)

        derivative_xi_1 = self.xi_k_true(c=c,
                                         d=2,
                                         psi_eigenvalues=self.psi_eigenvalues,
                                         shrinkage_list=self.par.plo.shrinkage_list,
                                         m_z_c=m_z_c)

        # psi_beta = np.sum(self.beta_eigenvalues * self.psi_eigenvalues)
        #
        # normalizer = self.par.simulated_data.noise_size ** 2 + psi_beta
        #
        # mean_true = [psi_beta - self.par.plo.shrinkage_list[i] * xi_beta_1[i] for i in
        #              range(len(self.par.plo.shrinkage_list))]
        #
        # xi_term = [xi_1[i] - self.par.plo.shrinkage_list[i] * deriplo
        #            for i in range(len(self.par.plo.shrinkage_list))]
        # xi_beta_term = [xi_beta_0[i] - self.par.plo.shrinkage_list[i] * derivative_xi_beta_0[i]
        #                 for i in range(len(self.par.plo.shrinkage_list))]
        #
        # denominator = 1 / (1 - c + c * self.par.plo.shrinkage_list * m_z_c)
        #
        # params = [denominator[i] * self.par.plo.shrinkage_list[i] for i in range(len(self.par.plo.shrinkage_list))]
        #
        # second_term = [(params[i] ** 2) * xi_beta_term[i]
        #                - self.par.plo.shrinkage_list[i] * xi_beta_1[i]
        #                for i in range(len(self.par.plo.shrinkage_list))]
        #
        # var_true = [normalizer * (mean_true[i] + second_term[i] + self.par.simulated_data.noise_size * xi_term[i])
        #             for i in range(len(self.par.plo.shrinkage_list))]

        # self.mean_true = mean_true
        # self.var_true = var_true

    def True_value_eq_176(self, data_type):

        if data_type == DataUsed.INS:
            data = self.features_ins
        if data_type == DataUsed.OOS:
            data = self.features_oos
        if data_type == DataUsed.TOTAL:
            data = self.features

        [T, P] = data.shape
        covariance = data.T @ data / T
        inverse = [np.linalg.pinv(covariance + z * np.eye(P)) for z in self.par.plo.shrinkage_list]
        true_value_sigma_beta_eq_176 = [np.trace(self.beta_eigenvalues * (self.psi_eigenvalues * i) @ covariance) for i
                                        in inverse]
        true_value_beta_eq_176 = [(self.beta_dict[0].reshape(1, -1) @ (self.psi_eigenvalues * i) @ covariance @
                                   self.beta_dict[0].reshape(-1, 1))[0] for i in inverse]
        self.true_value_sigma_beta_eq_176 = true_value_sigma_beta_eq_176
        self.true_value_beta_eq_176 = true_value_beta_eq_176

    @staticmethod
    def smart_eigenvalue_decomposition(features: np.ndarray):
        """
        Lemma 28: Efficient Eigen value decomposition
        :param features: features used to create covariance matrix times x P
        :param T: Weight used to normalize matrix
        :return: Left eigenvectors PxT and eigenvalues without zeros
        """
        [T, P] = features.shape

        if P > T:
            covariance = features @ features.T

        else:
            covariance = features.T @ features

        eigval, eigvec = np.linalg.eigh(covariance)
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]

        if P > T:
            # project features on normalized eigenvectors
            eigvec = features.T @ eigvec * (eigval ** (-1 / 2)).reshape(1, -1)

        eigval = eigval / T

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

        W = [projected_features.T @ x_ for x_ in stuff_divided]

        if P > T:
            # adjustment for zero eigen values
            cov_left_over = features @ features.T / T - projected_features.T @ projected_features
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
        ret_vec = [labels * pi[i] for i in range(len(shrinkage_list))]

        # Calculate strategy performance using insample dat
        estimator_perf = LeaveOut.estimator_performance(ret_vec, pi, labels_squared)

        # do an average over all pi_T_tau do get the \hat \pi estimator
        pi_avg = [np.mean(p) for p in pi]

        return estimator_perf, ret_vec, pi, pi_avg

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

        if type(beta_hat) == list:
            pi = [features_out_of_sample @ b for b in beta_hat]
            estimator_list = [p * labels_out_of_sample for p in pi]

        else:
            pi = features_out_of_sample @ beta_hat
            estimator_list = pi * labels_out_of_sample

        return LeaveOut.estimator_performance(estimator_list, pi, labels_squared), pi, estimator_list

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
        if type(estimator_list) == list:
            estimator_list_mean = [np.mean(e) for e in estimator_list]

            estimator_list_var = [np.mean(e ** 2) for e in estimator_list]

            estimator_list_std = [np.sqrt(estimator_list_var[i] - estimator_list_mean[i] ** 2)
                                  for i in range(len(estimator_list_var))]

            estimator_list_pi_2 = [np.mean(p ** 2) for p in pi]

            estimator_list_pi = [np.mean(p) for p in pi]

            estimator_list_sharpe = [estimator_list_mean[i] / estimator_list_std[i]
                                     for i in range(len(estimator_list))]

            estimator_list_mse = [labels_squared - 2 * estimator_list_mean[i] + estimator_list_pi_2[i]
                                  for i in range(len(estimator_list))]
        else:
            estimator_list_mean = np.mean(estimator_list)

            estimator_list_var = np.mean(estimator_list ** 2)

            estimator_list_std = np.sqrt(estimator_list_var - estimator_list_mean ** 2)
            estimator_list_pi_2 = np.mean(pi ** 2)

            estimator_list_pi = np.mean(pi)

            estimator_list_sharpe = estimator_list_mean / estimator_list_std

            estimator_list_mse = labels_squared - 2 * estimator_list_mean + estimator_list_pi_2

        estimator_perf = {}
        estimator_perf['mean'] = estimator_list_mean
        estimator_perf['var'] = estimator_list_var
        estimator_perf['std'] = estimator_list_std
        estimator_perf['pi_2'] = estimator_list_pi_2
        estimator_perf['mse'] = estimator_list_mse
        estimator_perf['sharpe'] = estimator_list_sharpe
        estimator_perf['pi'] = estimator_list_pi

        return estimator_perf

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
    def xi_k_true(c: float,
                  psi_eigenvalues: np.ndarray,
                  shrinkage_list: np.ndarray,
                  m_z_c: np.ndarray,
                  beta_eigenvalues: np.ndarray = None,
                  l: int = 1,
                  d: int = 1):

        denominator = 1 / (1 - c + c * shrinkage_list * m_z_c)
        params = [denominator[i] * shrinkage_list[i] for i in range(len(shrinkage_list))]

        inverse_eigen_values = [(psi_eigenvalues ** l) / ((psi_eigenvalues + p) ** d) for p in params]

        if beta_eigenvalues is None:
            p = psi_eigenvalues.shape[1]
            t = int(p * c)
            xi = [np.sum(inverse_eigen_values[i]) * denominator[i] / t for i in
                  range(len(shrinkage_list))]

        else:
            xi = [np.sum(beta_eigenvalues * inverse_eigen_values[i]) * denominator[i] for i in
                  range(len(shrinkage_list))]

        # if l ==1 and d == 1 :
        #     xi = denominator - 1

        # seems to work for full sample t and c adjusted very strange...
        # l = 0
        # d = 1
        #
        # c = c / train_frac
        #
        # psi_eigenvalues = loo.psi_eigenvalues
        #
        # denominator = 1 / (1 - c + c * shrinkage_list * m_z_c)
        # params = [denominator[i] * shrinkage_list[i] for i in range(len(shrinkage_list))]
        #
        # inverse_eigen_values = [(psi_eigenvalues ** l) / ((psi_eigenvalues + p) ** d) for p in params]
        #
        # p = psi_eigenvalues.shape[1]
        # xi = [1 + np.sum(inverse_eigen_values[i]) * denominator[i] / t for i in
        #       range(len(shrinkage_list))]
        #
        # est_m = [np.sum(inverse_eigen_values[i]) * denominator[i] / p for i in
        #          range(len(shrinkage_list))]

        return xi

    @staticmethod
    def optimal_weights(ret_vec_ins: list,
                        pi_ins: list):

        l = len(ret_vec_ins)
        ret_mat_for_z = np.array(ret_vec_ins).reshape(l, -1)

        # calculate average strategy return per unit of grid z
        v = np.sum(ret_mat_for_z, 1)

        pi_mat_for_z = np.array(pi_ins).reshape(l, -1)

        # average pi times pi for different combinations of the z grid
        m_mse = pi_mat_for_z @ pi_mat_for_z.T

        # average strategy return times strategy return for different combinations of the z grid
        m_sharpe = ret_mat_for_z @ ret_mat_for_z.T

        # weights for mse
        w_mse = np.linalg.pinv(m_mse) @ v

        # weights for sharpe
        w_sharpe = np.linalg.pinv(m_sharpe) @ v

        return w_sharpe, w_mse, m_sharpe, m_mse

    @staticmethod
    def optimal_shrinkage(w_sharpe: np.ndarray,
                          w_mse: np.ndarray,
                          beta_hat: list):

        l = len(beta_hat)
        beta_hat_mat = np.array(beta_hat).reshape(l, -1).T
        beta_hat_optimal_sharpe = (beta_hat_mat @ w_sharpe).reshape(-1, 1)
        beta_hat_optimal_mse = (beta_hat_mat @ w_mse).reshape(-1, 1)

        return beta_hat_optimal_sharpe, beta_hat_optimal_mse
